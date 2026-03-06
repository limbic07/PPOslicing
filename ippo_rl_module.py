import numpy as np
import torch
from ray.rllib.algorithms.ppo.torch.ppo_torch_rl_module import PPOTorchRLModule
from ray.rllib.core.rl_module.marl_module import SingleAgentRLModuleSpec

DEFAULT_INITIAL_SLICE_RATIOS = np.array([0.4, 0.4, 0.2], dtype=np.float32)
DEFAULT_INITIAL_ACTION_LOG_STD = -1.5


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


def build_initialized_rl_module_spec(
    action_softmax_temperature: float,
    *,
    fcnet_hiddens=None,
    fcnet_activation: str = "relu",
    initial_action_ratios=None,
    initial_action_log_std: float = DEFAULT_INITIAL_ACTION_LOG_STD,
) -> SingleAgentRLModuleSpec:
    return SingleAgentRLModuleSpec(
        module_class=InitializedPPOTorchRLModule,
        model_config_dict={
            "fcnet_hiddens": list(fcnet_hiddens or [256, 256]),
            "fcnet_activation": fcnet_activation,
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
        },
    )
