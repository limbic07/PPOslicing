import numpy as np
import torch
import torch.nn as nn
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core import Columns
from ray.rllib.core.models.base import ACTOR, ENCODER_OUT
from ray.rllib.core.rl_module.marl_module import SingleAgentRLModuleSpec

DEFAULT_INITIAL_SLICE_RATIOS = np.array([0.4, 0.4, 0.2], dtype=np.float32)
DEFAULT_INITIAL_ACTION_LOG_STD = -1.5
# 7 cells * 12 per-cell features (demand[3], queue[3], se[3], prev_ratio[3]).
CENTRALIZED_CRITIC_GLOBAL_DIM = 84


def resolve_local_obs_dim(observation_mode: str) -> int:
    mode = str(observation_mode).lower()
    if mode == "pure_local":
        return 14
    if mode == "neighbor_augmented":
        return 20
    raise ValueError(
        f"Unsupported observation_mode={observation_mode!r}. "
        "Expected one of ['pure_local', 'neighbor_augmented']"
    )


def _activation_layer(name: str):
    key = str(name).lower()
    if key == "relu":
        return nn.ReLU
    if key == "tanh":
        return nn.Tanh
    if key == "elu":
        return nn.ELU
    if key == "silu":
        return nn.SiLU
    raise ValueError(f"Unsupported activation={name!r}")


def ratios_to_raw_action_means(ratios, temperature: float) -> np.ndarray:
    ratios = np.asarray(ratios, dtype=np.float32)
    ratios = np.clip(ratios, 1e-8, None)
    ratios = ratios / np.sum(ratios)
    logits = np.log(ratios)
    logits = logits - np.mean(logits)
    return (logits / max(float(temperature), 1e-6)).astype(np.float32)


class InitializedPPOTorchRLModule(PPOTorchRLModule):
    """PPO torch RLModule with a deterministic initial action prior."""

    def setup(self):
        super().setup()

        model_cfg = self.config.model_config_dict or {}
        self.use_centralized_critic = bool(model_cfg.get("use_centralized_critic", False))
        self.actor_local_obs_dim = int(
            model_cfg.get(
                "actor_local_obs_dim",
                resolve_local_obs_dim(model_cfg.get("observation_mode", "pure_local")),
            )
        )
        self.critic_obs_dim = int(
            model_cfg.get(
                "critic_obs_dim",
                self.actor_local_obs_dim + (CENTRALIZED_CRITIC_GLOBAL_DIM if self.use_centralized_critic else 0),
            )
        )

        obs_space = self.config.observation_space
        if obs_space is None or not hasattr(obs_space, "shape"):
            raise ValueError("RLModule observation_space is required for CTDE masking.")
        self.module_obs_dim = int(obs_space.shape[0])
        if self.actor_local_obs_dim > self.module_obs_dim:
            raise ValueError(
                f"actor_local_obs_dim={self.actor_local_obs_dim} exceeds observation dim={self.module_obs_dim}"
            )
        if self.critic_obs_dim > self.module_obs_dim:
            raise ValueError(
                f"critic_obs_dim={self.critic_obs_dim} exceeds observation dim={self.module_obs_dim}"
            )

        if not self.config.inference_only and self.use_centralized_critic:
            vf_hiddens = list(model_cfg.get("vf_hiddens", model_cfg.get("fcnet_hiddens", [256, 256])))
            activation = _activation_layer(model_cfg.get("fcnet_activation", "relu"))
            layers = []
            in_dim = self.critic_obs_dim
            for hidden_dim in vf_hiddens:
                layers.append(nn.Linear(in_dim, int(hidden_dim)))
                layers.append(activation())
                in_dim = int(hidden_dim)
            self.ctde_vf_encoder = nn.Sequential(*layers) if layers else nn.Identity()
            self.ctde_vf_head = nn.Linear(in_dim, 1)

        target_ratios = model_cfg.get("initial_action_ratios", DEFAULT_INITIAL_SLICE_RATIOS)
        temperature = model_cfg.get("action_softmax_temperature", 3.0)
        initial_log_std = float(
            model_cfg.get("initial_action_log_std", DEFAULT_INITIAL_ACTION_LOG_STD)
        )
        target_means = ratios_to_raw_action_means(target_ratios, temperature)

        linear = self.pi.net.mlp[0]
        with torch.no_grad():
            linear.weight[:3].zero_()
            linear.bias[:3].copy_(torch.as_tensor(target_means, dtype=linear.bias.dtype))
            linear.weight[3:].zero_()
            linear.bias[3:].fill_(initial_log_std)

    def _masked_actor_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.actor_local_obs_dim >= self.module_obs_dim:
            return obs
        obs_actor = obs.clone()
        obs_actor[..., self.actor_local_obs_dim :] = 0.0
        return obs_actor

    def _critic_obs(self, obs: torch.Tensor) -> torch.Tensor:
        if self.critic_obs_dim >= self.module_obs_dim:
            return obs
        return obs[..., : self.critic_obs_dim]

    def _with_actor_obs(self, batch):
        actor_batch = dict(batch)
        actor_batch[Columns.OBS] = self._masked_actor_obs(batch[Columns.OBS])
        return actor_batch

    def _centralized_critic_values(self, obs: torch.Tensor) -> torch.Tensor:
        critic_obs = self._critic_obs(obs)
        hidden = self.ctde_vf_encoder(critic_obs)
        return self.ctde_vf_head(hidden).squeeze(-1)

    def _forward_inference(self, batch):
        output = {}
        actor_batch = self._with_actor_obs(batch)
        encoder_outs = self.encoder(actor_batch)
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]
        output[Columns.ACTION_DIST_INPUTS] = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        return output

    def _forward_exploration(self, batch, **kwargs):
        if not self.use_centralized_critic:
            return super()._forward_exploration(self._with_actor_obs(batch), **kwargs)

        if self.config.model_config_dict.get("uses_new_env_runners"):
            return self._forward_inference(batch)

        output = {}
        actor_batch = self._with_actor_obs(batch)
        encoder_outs = self.encoder(actor_batch)
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]

        if not self.config.inference_only:
            output[Columns.VF_PREDS] = self._centralized_critic_values(batch[Columns.OBS])

        output[Columns.ACTION_DIST_INPUTS] = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        return output

    def _forward_train(self, batch):
        if not self.use_centralized_critic:
            return super()._forward_train(self._with_actor_obs(batch))

        if self.config.inference_only:
            raise RuntimeError(
                "Trying to train a module that is not a learner module. "
                "Set inference_only=False when building the module."
            )

        output = {}
        actor_batch = self._with_actor_obs(batch)
        encoder_outs = self.encoder(actor_batch)
        if Columns.STATE_OUT in encoder_outs:
            output[Columns.STATE_OUT] = encoder_outs[Columns.STATE_OUT]

        output[Columns.VF_PREDS] = self._centralized_critic_values(batch[Columns.OBS])

        output[Columns.ACTION_DIST_INPUTS] = self.pi(encoder_outs[ENCODER_OUT][ACTOR])
        return output

    def compute_values(self, batch):
        if self.use_centralized_critic:
            return self._centralized_critic_values(batch[Columns.OBS])
        return super().compute_values(self._with_actor_obs(batch))


def build_initialized_rl_module_spec(
    action_softmax_temperature: float,
    *,
    fcnet_hiddens=None,
    fcnet_activation: str = "relu",
    initial_action_ratios=None,
    initial_action_log_std: float = DEFAULT_INITIAL_ACTION_LOG_STD,
    observation_mode: str = "pure_local",
    use_centralized_critic: bool = False,
    actor_local_obs_dim: int | None = None,
    critic_obs_dim: int | None = None,
    critic_global_dim: int = CENTRALIZED_CRITIC_GLOBAL_DIM,
) -> SingleAgentRLModuleSpec:
    resolved_actor_local_obs_dim = (
        int(actor_local_obs_dim)
        if actor_local_obs_dim is not None
        else resolve_local_obs_dim(observation_mode)
    )
    resolved_critic_obs_dim = (
        int(critic_obs_dim)
        if critic_obs_dim is not None
        else resolved_actor_local_obs_dim + (int(critic_global_dim) if use_centralized_critic else 0)
    )
    return SingleAgentRLModuleSpec(
        module_class=InitializedPPOTorchRLModule,
        model_config_dict={
            "fcnet_hiddens": list(fcnet_hiddens or [256, 256]),
            "fcnet_activation": fcnet_activation,
            "vf_hiddens": list(fcnet_hiddens or [256, 256]),
            "action_softmax_temperature": float(action_softmax_temperature),
            "initial_action_ratios": list(
                np.asarray(
                    initial_action_ratios
                    if initial_action_ratios is not None
                    else DEFAULT_INITIAL_SLICE_RATIOS,
                    dtype=np.float32,
                )
            ),
            "initial_action_log_std": float(initial_action_log_std),
            "observation_mode": str(observation_mode),
            "use_centralized_critic": bool(use_centralized_critic),
            "actor_local_obs_dim": int(resolved_actor_local_obs_dim),
            "critic_obs_dim": int(resolved_critic_obs_dim),
        },
    )
