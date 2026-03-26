import numpy as np
from collections import deque

def simulate_optimal_cooperative_policy(steps=5000, seed=42):
    np.random.seed(seed)
    
    # --- Environment Constants (Balanced Profile) ---
    NUM_CELLS = 7
    TTI = 0.0005  # 0.5 ms
    BW_MHZ = 100.0
    
    # Traffic Params
    EMBB_MEAN, EMBB_STD = 250.0, 40.0
    URLLC_NOMINAL_MEAN, URLLC_NOMINAL_STD = 10.0, 2.0
    URLLC_BURST_MEAN, URLLC_BURST_STD = 100.0, 15.0
    URLLC_BURST_START, URLLC_BURST_END = 0.06, 0.35
    MMTC_MEAN, MMTC_STD = 10.0, 1.0
    
    # SE AR(1) Params
    SE_MEAN = np.array([4.5, 2.5, 1.5], dtype=np.float32)
    SE_MIN = np.array([2.0, 1.0, 0.5], dtype=np.float32)
    SE_MAX = np.array([6.0, 4.0, 2.5], dtype=np.float32)
    RHO = 0.9
    NOISE_STD = np.array([0.4, 0.2, 0.15], dtype=np.float32)
    
    # SLA Params
    EMBB_GBR = 200.0
    URLLC_MAX_DELAY = 0.002
    MMTC_MAX_QUEUE = 1.0
    WINDOW_SIZE = 20
    
    # ICI Params
    ICI_GAIN = 0.50
    SE_FLOOR = 0.45
    
    # Topology Map (0 is center, 1-6 are edges)
    neighbor_map = {
        0: [1, 2, 3, 4, 5, 6],
        1: [0, 2, 6], 2: [0, 1, 3], 3: [0, 2, 4],
        4: [0, 3, 5], 5: [0, 4, 6], 6: [0, 5, 1]
    }
    
    # State tracking
    queues = np.zeros((NUM_CELLS, 3), dtype=np.float32)
    current_se = np.tile(SE_MEAN, (NUM_CELLS, 1))
    burst_state = np.zeros(NUM_CELLS, dtype=bool)
    embb_history = {i: deque(maxlen=WINDOW_SIZE) for i in range(NUM_CELLS)}
    
    # Metrics
    violations = np.zeros((NUM_CELLS, 3), dtype=int)
    total_steps = 0
    
    print(f"Running Ideal Cooperative Oracle Simulation for {steps} TTIs...")
    
    for step in range(steps):
        total_steps += 1
        arrivals = np.zeros((NUM_CELLS, 3), dtype=np.float32)
        
        # 1. Update State & Traffic
        for i in range(NUM_CELLS):
            # SE Evolution
            noise = np.random.normal(0, NOISE_STD)
            current_se[i] = RHO * current_se[i] + (1 - RHO) * SE_MEAN + np.sqrt(1 - RHO**2) * noise
            current_se[i] = np.clip(current_se[i], SE_MIN, SE_MAX)
            
            # Traffic Arrivals
            arr_embb = np.clip(np.random.normal(EMBB_MEAN, EMBB_STD), 180.0, 350.0)
            
            if burst_state[i]:
                if np.random.rand() < URLLC_BURST_END: burst_state[i] = False
            else:
                if np.random.rand() < URLLC_BURST_START: burst_state[i] = True
                
            if burst_state[i]:
                arr_urllc = max(0.0, np.random.normal(URLLC_BURST_MEAN, URLLC_BURST_STD))
            else:
                arr_urllc = max(0.0, np.random.normal(URLLC_NOMINAL_MEAN, URLLC_NOMINAL_STD))
                
            arr_mmtc = np.random.normal(MMTC_MEAN, MMTC_STD)
            
            arrivals[i] = [arr_embb, arr_urllc, arr_mmtc]
            queues[i] += arrivals[i] * TTI
            
        # 2. Optimal Cooperative Policy (Two-Pass Heuristic)
        actions = np.zeros((NUM_CELLS, 3), dtype=np.float32)
        
        # Pass 1: Estimate required ratio assuming NO ICI
        req_ratios = np.zeros((NUM_CELLS, 3), dtype=np.float32)
        for i in range(NUM_CELLS):
            # Target to empty queue in this TTI
            target_mbps = queues[i] / TTI
            # Required ratio = Target Mbps / (BW * SE)
            req_ratios[i] = target_mbps / (BW_MHZ * current_se[i])
            req_ratios[i] = np.clip(req_ratios[i], 0.0, 1.0)
            
        # Pass 2: Calculate actual ICI based on neighbor's requirements, then finalize actions
        for i in range(NUM_CELLS):
            neighbors = neighbor_map[i]
            neighbor_reqs = req_ratios[neighbors]
            # normalized_neighbor_load
            n_load = np.sum(neighbor_reqs, axis=0) / len(neighbors)
            se_modifier = np.clip(1.0 - (ICI_GAIN * n_load), SE_FLOOR, 1.0)
            
            effective_se = current_se[i] * se_modifier
            target_mbps = queues[i] / TTI
            
            # Recalculate exact ratios needed
            final_req = target_mbps / (BW_MHZ * effective_se)
            
            # Priority Allocation: URLLC -> mMTC -> eMBB gets the rest
            bw_u = min(final_req[1], 1.0)
            bw_m = min(final_req[2], 1.0 - bw_u)
            bw_e = max(0.0, 1.0 - bw_u - bw_m) # eMBB takes ALL remaining bandwidth
            
            actions[i] = [bw_e, bw_u, bw_m]
            
        # 3. Environment Step (Process Service)
        for i in range(NUM_CELLS):
            neighbors = neighbor_map[i]
            neighbor_actions = actions[neighbors]
            n_load = np.sum(neighbor_actions, axis=0) / len(neighbors)
            se_modifier = np.clip(1.0 - (ICI_GAIN * n_load), SE_FLOOR, 1.0)
            
            effective_se = current_se[i] * se_modifier
            service_rate_mbps = actions[i] * BW_MHZ * effective_se
            service_capacity_mb = service_rate_mbps * TTI
            
            served_mb = np.minimum(service_capacity_mb, queues[i])
            queues[i] -= served_mb
            
            # eMBB History
            achieved_tp_mbps = served_mb / TTI
            embb_history[i].append(achieved_tp_mbps[0])
            
            # SLA Evaluation
            if len(embb_history[i]) == WINDOW_SIZE:
                embb_avg = np.mean(embb_history[i])
                if embb_avg < EMBB_GBR: violations[i, 0] += 1
                
            # URLLC Delay estimation
            safe_rate = max(service_rate_mbps[1], 0.1)
            est_delay = queues[i][1] / safe_rate
            if est_delay > URLLC_MAX_DELAY: violations[i, 1] += 1
            
            # mMTC Queue
            if queues[i][2] > MMTC_MAX_QUEUE: violations[i, 2] += 1
            queues[i][2] = min(queues[i][2], MMTC_MAX_QUEUE)

    # --- Print Results ---
    eval_steps = total_steps - WINDOW_SIZE + 1
    print("\n" + "="*50)
    print("Ideal Oracle Policy SLA Success Rates:")
    print("="*50)
    for i in range(NUM_CELLS):
        label = "BS_0 (Center)" if i == 0 else f"BS_{i} (Edge)"
        succ_e = 100.0 * (1.0 - violations[i, 0] / eval_steps)
        succ_u = 100.0 * (1.0 - violations[i, 1] / eval_steps)
        succ_m = 100.0 * (1.0 - violations[i, 2] / eval_steps)
        print(f"{label:<15} | eMBB: {succ_e:6.2f}% | URLLC: {succ_u:6.2f}% | mMTC: {succ_m:6.2f}%")
        
    global_succ = 100.0 * (1.0 - np.sum(violations, axis=0) / (NUM_CELLS * eval_steps))
    print("-" * 50)
    print(f"Global Average  | eMBB: {global_succ[0]:6.2f}% | URLLC: {global_succ[1]:6.2f}% | mMTC: {global_succ[2]:6.2f}%")

if __name__ == "__main__":
    simulate_optimal_cooperative_policy(steps=5000)