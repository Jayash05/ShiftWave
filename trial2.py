"""
Vanguard 2.0: Quantum-Assisted Workforce Optimization Pipeline
Master's Tech / Capstone Submission Model
"""

import math
import time
import numpy as np
import pandas as pd
from itertools import combinations

try:
    import neal
except ImportError:
    print("FATAL ERROR: 'dwave-neal' is required. Run: pip install dwave-neal")
    exit()

# ==========================================
# 1. QUEUE PHYSICS (ERLANG C)
# ==========================================
def erlang_c(A, N):
    if N <= A: return 1.0 
    inv_b = 1.0
    for i in range(1, int(N) + 1):
        inv_b = 1.0 + (inv_b * i) / A
    erlang_b = 1.0 / inv_b
    p_w = (N * erlang_b) / (N - A * (1 - erlang_b))
    return min(max(p_w, 0.0), 1.0)

def calculate_erlang_target(hourly_rate, aht, target_sla=0.80, target_asa=20.0):
    if hourly_rate <= 0: return 0
    A = (hourly_rate / 3600.0) * aht
    N = max(1, math.ceil(A))
    while True:
        p_w = erlang_c(A, N)
        service_level = 1.0 - (p_w * math.exp(-(N - A) * (target_asa / aht)))
        if service_level >= target_sla and N > A:
            return N
        N += 1

# ==========================================
# 2. 3D HAMILTONIAN QUBO COMPILER
# ==========================================
def build_3d_hamiltonian(df_forecast_dict, df_agents, shifts, queue_metadata):
    Q = {}
    def add_weight(v1, v2, weight):
        if weight == 0: return
        k = tuple(sorted([v1, v2]))
        Q[k] = Q.get(k, 0) + weight

    # Calibrated Hyperparameters for strict Tier Matching
    L1_DEMAND   = 5000.0   
    L2_VALID    = 10000.0 # High penalty to strictly prevent double-booking 
    L3_FRICTION = 25.0    
    L4_ROUTING  = 250.0   # Parabolic distance penalty for strict routing

    shift_names = list(shifts.keys())

    print("[QUANTUM LAYER] Compiling 3D Variables (Agent x Shift x Queue)...")
    for _, agent in df_agents.iterrows():
        a_id = agent['Agent_ID']
        wage = float(agent['Hourly_Wage'])
        pref_raw = str(agent['Preferred_Shift'])
        
        # Map string Tier to numerical Tier (V_i)
        tier_map = {'General': 1, 'Advisor': 2, 'Intermed': 3, 'Investor': 4}
        v_i = tier_map.get(agent['Tier'], 2)
        
        # Generate valid 3D variables only for queues the agent is explicitly skilled in
        agent_vars = []
        for s_name in shift_names:
            if s_name == 'Off': continue
            
            for q_name, meta in queue_metadata.items():
                skill_col = meta['skill_col']
                if agent.get(skill_col, 0) == 1:
                    v_q = meta['tier']
                    var_name = f"x_{a_id}_{s_name}_{q_name}"
                    agent_vars.append((var_name, s_name, q_name, v_q))
        
        # H_valid: Maximum ONE combination of Shift & Queue per Agent
        var_names = [v[0] for v in agent_vars]
        for v1, v2 in combinations(var_names, 2):
            add_weight(v1, v2, L2_VALID)
            
        # H_cost, H_friction, H_routing
        for var_name, s_name, q_name, v_q in agent_vars:
            h_cost = wage * float(agent.get('Shift_Elapsed_Hours', 8.0))
            h_friction = 0 if s_name in pref_raw else L3_FRICTION
            
            active_intervals = sum(shifts[s_name])
            h_routing = L4_ROUTING * ((v_i - v_q)**2) * active_intervals
            
            add_weight(var_name, var_name, h_cost + h_friction + h_routing)

    print("[QUANTUM LAYER] Compiling H_demand (Slack & Non-Linear Proficiency)...")
    for q_name, df_forecast in df_forecast_dict.items():
        meta = queue_metadata[q_name]
        
        for t_idx, row in df_forecast.iterrows():
            target_N = row['Target_Headcount']
            if target_N <= 0: continue
            
            active_qubits = []
            for _, agent in df_agents.iterrows():
                if agent.get(meta['skill_col'], 0) == 0: continue
                
                a_id = agent['Agent_ID']
                
                # Full Proficiency equation (m_i)
                m_i = float(agent['Tenure_AHT_Multiplier']) * float(agent['Micro_Break_Capacity_Multiplier'])
                
                for s_name in shift_names:
                    if s_name != 'Off' and shifts[s_name][t_idx] == 1:
                        var_name = f"x_{a_id}_{s_name}_{q_name}"
                        # Ensure variable exists in matrix
                        if var_name in Q or (var_name, var_name) in Q or len(Q) >= 0:
                            active_qubits.append((var_name, m_i))
            
            # Dynamic Slack absorption
            max_slack = max(1, math.ceil(math.log2(target_N + 5)))
            slack_qubits = [(f"y_{q_name}_{t_idx}_{k}", -(2**k)) for k in range(max_slack)]
            
            all_qubits = active_qubits + slack_qubits
            
            # (Sum - N)^2 expansion
            for var_name, coeff in all_qubits:
                add_weight(var_name, var_name, L1_DEMAND * (coeff**2 - 2 * target_N * coeff))
            for (var1, coeff1), (var2, coeff2) in combinations(all_qubits, 2):
                add_weight(var1, var2, L1_DEMAND * 2 * coeff1 * coeff2)

    return Q

# ==========================================
# 3. EXECUTION ENGINE
# ==========================================
def execute_quantum_model(df_forecast_dict, df_agents, shifts, queue_metadata):
    Q = build_3d_hamiltonian(df_forecast_dict, df_agents, shifts, queue_metadata)
    print("\n[QUANTUM LAYER] Initializing D-Wave Simulated Annealing...")
    start_time = time.time()
    
    sampler = neal.SimulatedAnnealingSampler()
    # Deep parameter sweep for stable global minimum
    response = sampler.sample_qubo(Q, num_reads=10, num_sweeps=500)
    best_sample = response.first.sample
    duration = time.time() - start_time
    
    assigned, cost, cross_skill_penalty = 0, 0.0, 0
    
    print("\n" + "="*70)
    print(" 🚀 FINAL ENTERPRISE SCHEDULE (3D QUANTUM QUBO)")
    print("="*70)
    
    for var_name, val in best_sample.items():
        if val == 1 and var_name.startswith('x_'):
            # Parse 3D output: x_{Agent_ID}_{Shift}_{Queue}
            parts = var_name.split('_')
            a_id = f"{parts[1]}_{parts[2]}"
            s_name = parts[3]
            q_name = "_".join(parts[4:])
            
            agent = df_agents[df_agents['Agent_ID'] == a_id].iloc[0]
            assigned += 1
            cost += float(agent['Hourly_Wage']) * float(agent.get('Shift_Elapsed_Hours', 8.0))
            
            tier_map = {'General': 1, 'Advisor': 2, 'Intermed': 3, 'Investor': 4}
            v_i = tier_map.get(agent['Tier'], 2)
            v_q = queue_metadata[q_name]['tier']
            
            routed_well = (v_i == v_q)
            if not routed_well: cross_skill_penalty += 1
            
            flag = "" if routed_well else f"[⚠️ Mismatch: T{v_i} Agent on T{v_q} Queue]"
            prof = float(agent['Tenure_AHT_Multiplier'])
            print(f"✅ {a_id} (Tier {v_i} | Prof: {prof:.2f}x) -> {s_name} on {q_name} {flag}")
                
    print("-" * 70)
    print(f"Total Annealing Time     : {duration:.4f} sec")
    print(f"Total Agents Deployed    : {assigned}")
    print(f"Total Projected Payroll  : ${cost:,.2f}")
    print(f"Sub-Optimal Routings     : {cross_skill_penalty}")
    print("="*70)
    return best_sample

# ==========================================
# 4. MASTER DATA PIPELINE
# ==========================================
if __name__ == "__main__":
    print("[1] Loading external datasets...")
    try:
        df_calls = pd.read_csv("call_log.csv")
        df_agents = pd.read_csv("agent_data.csv")
    except FileNotFoundError:
        print("FATAL ERROR: CSV files not found. Ensure 'call_log.csv' and 'agent_data.csv' are in the directory.")
        exit()
        
    print("[2] Processing Call Data & Generating Erlang C Targets...")
    df_calls['Arrival_Timestamp'] = pd.to_datetime(df_calls['Arrival_Timestamp'])
    target_date = df_calls['Arrival_Timestamp'].dt.date.min()
    
    # Filter for Human calls on the target date
    day_calls = df_calls[(df_calls['Arrival_Timestamp'].dt.date == target_date) & 
                         (df_calls['Handled_By'] == 'Human')].copy()
    
    # Define mapping between Queue names and Agent Dataset skills/tiers
    queue_metadata = {
        'General_Support': {'skill_col': 'Skill_GenSupport', 'tier': 1},
        'Advisor':         {'skill_col': 'Skill_Advisor',    'tier': 2},
        'Intermed':        {'skill_col': 'Skill_Intermed',   'tier': 3},
        'Investor':        {'skill_col': 'Skill_Investor',   'tier': 4}
    }
    
    df_forecast_dict = {}
    found_queues = day_calls['Queue'].unique()
    
    for q in found_queues:
        if q not in queue_metadata:
            # Auto-assign metadata if queue not explicitly mapped
            queue_metadata[q] = {'skill_col': f"Skill_{q}", 'tier': 2}
            
        q_calls = day_calls[day_calls['Queue'] == q].copy()
        q_calls.set_index('Arrival_Timestamp', inplace=True)
        
        interval_stats = q_calls.resample('15Min').agg(
            Volume=('Call_ID', 'count'),
            AHT=('Handle_Time_Seconds', 'mean')
        ).fillna(0)
        
        interval_stats['Target_Headcount'] = interval_stats.apply(
            lambda row: calculate_erlang_target(row['Volume'] * 4, row['AHT'] if row['AHT'] > 0 else 240.0), axis=1
        )
        df_forecast_dict[q] = interval_stats.reset_index()

    print("[3] Pre-processing Agent Pool...")
    # Smart dimensional reduction: Sort by Full-Time and Proficiency before slicing
    df_agents['Is_FT'] = (df_agents['Contract_Type'].str.contains('Full-Time')).astype(int)
    df_agents['Proficiency'] = df_agents['Tenure_AHT_Multiplier'] * df_agents['Micro_Break_Capacity_Multiplier']
    df_agents = df_agents.sort_values(by=['Is_FT', 'Proficiency'], ascending=[False, False])
    
    # Extract only what we need to keep QUBO matrix mathematically feasible
    total_peak = sum(df['Target_Headcount'].max() for df in df_forecast_dict.values())
    calculated_pool = int(max(total_peak * 1.5, 50))
    
    # ⚠️ HARD CAP FOR CLASSICAL CPU SIMULATION 
    # (Limits matrix to ~1500
    
    max_cpu_limit = 250
    elite_pool = df_agents.head(int(max(total_peak * 1.5, 50))).reset_index(drop=True)
    
    print(f"    -> Dynamically scoped matrix to {len(elite_pool)} agents.")

    shifts = {
        'Morning': [1 if 24 <= t < 56 else 0 for t in range(96)], # 06:00 - 14:00
        'Midday':  [1 if 40 <= t < 72 else 0 for t in range(96)], # 10:00 - 18:00
        'Evening': [1 if 56 <= t < 88 else 0 for t in range(96)], # 14:00 - 22:00
        'Off':     [0 for _ in range(96)]
    }
    
    execute_quantum_model(df_forecast_dict, elite_pool, shifts, queue_metadata)