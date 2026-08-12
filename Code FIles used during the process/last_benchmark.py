import math
import time
import numpy as np
import pandas as pd
from itertools import combinations
import pulp

try:
    import neal
except ImportError:
    print("FATAL ERROR: 'dwave-neal' is required. Run: pip install dwave-neal")
    exit()

# 1. QUEUE PHYSICS (ERLANG C)
def erlang_c(A, N):
    if N <= A: return 1.0 
    inv_b = 1.0
    for i in range(1, int(N) + 1):
        inv_b = 1.0 + (inv_b * i) / A
    erlang_b = 1.0 / inv_b
    p_w = (N * erlang_b) / (N - A * (1 - erlang_b))
    return min(max(p_w, 0.0), 1.0)

def calculate_erlang_target(hourly_rate, aht, target_sla=0.90, target_asa=10.0):
    if hourly_rate <= 0: return 0
    A = (hourly_rate / 3600.0) * aht
    N = max(1, math.ceil(A))
    while True:
        p_w = erlang_c(A, N)
        service_level = 1.0 - (p_w * math.exp(-(N - A) * (target_asa / aht)))
        if service_level >= target_sla and N > A:
            return N
        N += 1


# 2. EVALUATION ENGINE (SLA & QUEUE METRICS)
def evaluate_schedules(df_forecast_dict, best_sample, df_agents, shifts):
    schedule_counts = {q: {t: 0 for t in range(96)} for q in df_forecast_dict.keys()}
    off_pref = 0
    
    for var_name, val in best_sample.items():
        if val == 1 and var_name.startswith('x_'):
            parts = var_name.split('_')
            a_id = f"{parts[1]}_{parts[2]}"
            s_name = parts[3]
            q_name = "_".join(parts[4:])
            
            # [FIX:] If Quantum noise assigns an agent to a queue with no forecast to safely track it
            if q_name not in schedule_counts:
                schedule_counts[q_name] = {t: 0 for t in range(96)}
                
            agent = df_agents[df_agents['Agent_ID'] == a_id].iloc[0]
            if s_name not in str(agent['Preferred_Shift']):
                off_pref += 1
            
            m_i = float(agent['Tenure_AHT_Multiplier']) * float(agent['Micro_Break_Capacity_Multiplier'])
            for t_idx in range(96):
                if shifts[s_name][t_idx] == 1:
                    schedule_counts[q_name][t_idx] += m_i
                    
    total_calls, total_handled, total_sl_calls, total_wait_sec, total_abandoned = 0, 0, 0, 0, 0
    
    for q_name, df_forecast in df_forecast_dict.items():
        for t_idx, row in df_forecast.iterrows():
            vol = row.get('Volume', row['Target_Headcount'])
            aht = row.get('AHT', 240.0)
            if vol <= 0: continue
            
            total_calls += vol
            interval_capacity = schedule_counts[q_name][t_idx]
            A = (vol / 900.0) * aht
            
            if interval_capacity > A and interval_capacity > 0:
                pw = erlang_c(A, interval_capacity)
                asa = (pw * aht) / (interval_capacity - A)
                sl = 1.0 - (pw * math.exp(-(interval_capacity - A) * (20.0 / max(aht, 1))))
                sl = min(max(sl, 0.0), 1.0)
                
                abandon_rate = pw * math.exp(-300.0 / max(asa, 1.0))
                abandoned = int(vol * abandon_rate)
                answered = int(vol - abandoned)
                
                total_abandoned += abandoned
                total_handled += answered
                total_wait_sec += answered * asa
                total_sl_calls += answered * sl
            else:
                abandoned = int(vol * 0.8)
                total_abandoned += abandoned
                total_handled += (vol - abandoned)
                total_wait_sec += (vol - abandoned) * 450.0
                
    sla_pct = (total_sl_calls / max(total_handled, 1)) * 100
    asa = total_wait_sec / max(total_handled, 1)
    abandon_pct = (total_abandoned / max(total_calls, 1)) * 100
    
    return sla_pct, asa, abandon_pct, off_pref


# 3. CLASSICAL LP BASELINE (PuLP)
def execute_classical_model(df_forecast_dict, df_agents, shifts, queue_metadata):
    print("\n[CLASSICAL LAYER] Executing PuLP Linear Programming (Up to 5 min limit)...")
    start_time = time.time()
    prob = pulp.LpProblem("WFM_Classical", pulp.LpMinimize)

    x_vars = {}
    for _, agent in df_agents.iterrows():
        a_id = agent['Agent_ID']
        for s_name in shifts.keys():
            if s_name == 'Off': continue
            for q_name, meta in queue_metadata.items():
                if agent.get(meta['skill_col'], 0) == 1:
                    x_vars[(a_id, s_name, q_name)] = pulp.LpVariable(f"x_{a_id}_{s_name}_{q_name}", cat='Binary')

    for _, agent in df_agents.iterrows():
        a_id = agent['Agent_ID']
        agent_vars = [x_vars[(a_id, s_name, q_name)] for s_name in shifts.keys() if s_name != 'Off' for q_name, meta in queue_metadata.items() if agent.get(meta['skill_col'], 0) == 1]
        if agent_vars: prob += pulp.lpSum(agent_vars) <= 1

    for q_name, df_forecast in df_forecast_dict.items():
        meta = queue_metadata[q_name]
        for t_idx, row in df_forecast.iterrows():
            target_N = row['Target_Headcount']
            if target_N <= 0: continue
            
            active_vars = []
            for _, agent in df_agents.iterrows():
                if agent.get(meta['skill_col'], 0) == 1:
                    a_id = agent['Agent_ID']
                    m_i = float(agent['Tenure_AHT_Multiplier']) * float(agent['Micro_Break_Capacity_Multiplier'])
                    for s_name in shifts.keys():
                        if s_name != 'Off' and shifts[s_name][t_idx] == 1:
                            active_vars.append(m_i * x_vars[(a_id, s_name, q_name)])
            
            if active_vars: prob += pulp.lpSum(active_vars) >= target_N

    L3_FRICTION, L4_ROUTING = 25.0, 250.0
    objective = []
    
    for _, agent in df_agents.iterrows():
        a_id = agent['Agent_ID']
        wage = float(agent['Hourly_Wage'])
        pref = str(agent['Preferred_Shift'])
        v_i = {'General': 1, 'Advisor': 2, 'Intermed': 3, 'Investor': 4}.get(agent.get('Tier', 'Advisor'), 2)
        
        for s_name in shifts.keys():
            if s_name == 'Off': continue
            for q_name, meta in queue_metadata.items():
                if agent.get(meta['skill_col'], 0) == 1:
                    v_q = meta['tier']
                    h_cost = wage * float(agent.get('Shift_Elapsed_Hours', 8.0))
                    h_friction = 0 if s_name in pref else L3_FRICTION
                    h_routing = L4_ROUTING * ((v_i - v_q)**2) * sum(shifts[s_name])
                    objective.append((h_cost + h_friction + h_routing) * x_vars[(a_id, s_name, q_name)])

    prob += pulp.lpSum(objective)
    
    # 5 MINUTE MAX TIMEOUT FOR CLASSICAL SOLVER
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=300))
    duration = time.time() - start_time

    best_sample = {}
    assigned, cost, cross_skill = 0, 0.0, 0
    
    for (a_id, s_name, q_name), var in x_vars.items():
        val = pulp.value(var)
        if val and val > 0.5:
            best_sample[f"x_{a_id}_{s_name}_{q_name}"] = 1
            assigned += 1
            agent = df_agents[df_agents['Agent_ID'] == a_id].iloc[0]
            cost += float(agent['Hourly_Wage']) * float(agent.get('Shift_Elapsed_Hours', 8.0))
            v_i = {'General': 1, 'Advisor': 2, 'Intermed': 3, 'Investor': 4}.get(agent.get('Tier', 'Advisor'), 2)
            if v_i != queue_metadata[q_name]['tier']: cross_skill += 1
        else:
            best_sample[f"x_{a_id}_{s_name}_{q_name}"] = 0

    return best_sample, duration, assigned, cost, cross_skill

# 4. QUANTUM 3D QUBO
def build_3d_hamiltonian(df_forecast_dict, df_agents, shifts, queue_metadata):
    Q = {}
    def add_weight(v1, v2, weight):
        if weight == 0: return
        k = tuple(sorted([v1, v2]))
        Q[k] = Q.get(k, 0) + weight

    L1_DEMAND   = 25000.0   
    L2_VALID    = 100000.0 
    L3_FRICTION = 1500.0    
    L4_ROUTING  = 2500.0   
    cost_weight = 0.05

    shift_names = list(shifts.keys())

    print("[QUANTUM LAYER] Compiling 3D Variables (Agent x Shift x Queue)...")
    for _, agent in df_agents.iterrows():
        a_id = agent['Agent_ID']
        wage = float(agent['Hourly_Wage'])
        pref_raw = str(agent['Preferred_Shift'])
        
        tier_map = {'General': 1, 'Advisor': 2, 'Intermed': 3, 'Investor': 4}
        v_i = tier_map.get(agent['Tier'], 2)
        
        agent_vars = []
        for s_name in shift_names:
            if s_name == 'Off': continue
            for q_name, meta in queue_metadata.items():
                skill_col = meta['skill_col']
                if agent.get(skill_col, 0) == 1:
                    v_q = meta['tier']
                    var_name = f"x_{a_id}_{s_name}_{q_name}"
                    agent_vars.append((var_name, s_name, q_name, v_q))
        
        var_names = [v[0] for v in agent_vars]
        for v1, v2 in combinations(var_names, 2):
            add_weight(v1, v2, L2_VALID)
            
        for var_name, s_name, q_name, v_q in agent_vars:
            h_cost = wage * float(agent.get('Shift_Elapsed_Hours', 8.0)) * cost_weight
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
                m_i = float(agent['Tenure_AHT_Multiplier']) * float(agent['Micro_Break_Capacity_Multiplier'])
                
                for s_name in shift_names:
                    if s_name != 'Off' and shifts[s_name][t_idx] == 1:
                        var_name = f"x_{a_id}_{s_name}_{q_name}"
                        if var_name in Q or (var_name, var_name) in Q or len(Q) >= 0:
                            active_qubits.append((var_name, m_i))
            
            max_slack = max(1, math.ceil(math.log2(target_N + 5)))
            slack_qubits = [(f"y_{q_name}_{t_idx}_{k}", -(2**k)) for k in range(max_slack)]
            all_qubits = active_qubits + slack_qubits
            
            for var_name, coeff in all_qubits:
                add_weight(var_name, var_name, L1_DEMAND * (coeff**2 - 2 * target_N * coeff))
            for (var1, coeff1), (var2, coeff2) in combinations(all_qubits, 2):
                add_weight(var1, var2, L1_DEMAND * 2 * coeff1 * coeff2)

    return Q

def execute_quantum_model(df_forecast_dict, df_agents, shifts, queue_metadata):
    Q = build_3d_hamiltonian(df_forecast_dict, df_agents, shifts, queue_metadata)
    print("\n[QUANTUM LAYER] Initializing D-Wave Simulated Annealing...")
    start_time = time.time()
    
    sampler = neal.SimulatedAnnealingSampler()
    # Increased sweeps for full enterprise stability
    response = sampler.sample_qubo(Q, num_reads=10, num_sweeps=1500)
    best_sample = response.first.sample
    duration = time.time() - start_time
    
    assigned, cost, cross_skill_penalty = 0, 0.0, 0
    for var_name, val in best_sample.items():
        if val == 1 and var_name.startswith('x_'):
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
            if v_i != v_q: cross_skill_penalty += 1
                
    return best_sample, duration, assigned, cost, cross_skill_penalty

# 5. MASTER EXECUTION & SCORECARD
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
    day_calls = df_calls[(df_calls['Arrival_Timestamp'].dt.date == target_date) & (df_calls['Handled_By'] == 'Human')].copy()
    
    queue_metadata = {
        'General_Support': {'skill_col': 'Skill_GenSupport', 'tier': 1},
        'Advisor':         {'skill_col': 'Skill_Advisor',    'tier': 2},
        'Intermed':        {'skill_col': 'Skill_Intermed',   'tier': 3},
        'Investor':        {'skill_col': 'Skill_Investor',   'tier': 4}
    }
    
    df_forecast_dict = {}
    for q in day_calls['Queue'].unique():
        if q not in queue_metadata: queue_metadata[q] = {'skill_col': f"Skill_{q}", 'tier': 2}
            
        q_calls = day_calls[day_calls['Queue'] == q].copy()
        q_calls.set_index('Arrival_Timestamp', inplace=True)
        
        interval_stats = q_calls.resample('15Min').agg(Volume=('Call_ID', 'count'), AHT=('Handle_Time_Seconds', 'mean')).fillna(0)
        interval_stats['Target_Headcount'] = interval_stats.apply(
            lambda row: calculate_erlang_target(row['Volume'] * 4, row['AHT'] if row['AHT'] > 0 else 240.0), axis=1)
        df_forecast_dict[q] = interval_stats.reset_index()

    print("[3] Pre-processing Agent Pool...")
    df_agents['Is_FT'] = (df_agents['Contract_Type'].str.contains('Full-Time')).astype(int)
    df_agents['Proficiency'] = df_agents['Tenure_AHT_Multiplier'] * df_agents['Micro_Break_Capacity_Multiplier']
    df_agents = df_agents.sort_values(by=['Is_FT', 'Proficiency'], ascending=[False, False])
    
    #Passing the full mathematically required matrix.
    total_peak = sum(df['Target_Headcount'].max() for df in df_forecast_dict.values())
    calculated_pool = int(max(total_peak * 1.5, 50))
    elite_pool = df_agents.head(calculated_pool).reset_index(drop=True)
    
    print(f" Target Demand requires ~{calculated_pool} agents.")
    print(f"Dynamically scoped matrix to {len(elite_pool)} agents (FULL POOL UNCAPPED).")

    shifts = {
        'Morning': [1 if 24 <= t < 56 else 0 for t in range(96)],
        'Midday':  [1 if 40 <= t < 72 else 0 for t in range(96)],
        'Evening': [1 if 56 <= t < 88 else 0 for t in range(96)],
        'Off':     [0 for _ in range(96)]
    }
    
    c_sample, c_time, c_assigned, c_cost, c_cross = execute_classical_model(df_forecast_dict, elite_pool, shifts, queue_metadata)
    q_sample, q_time, q_assigned, q_cost, q_cross = execute_quantum_model(df_forecast_dict, elite_pool, shifts, queue_metadata)
    
    c_sla, c_asa, c_aband, c_off = evaluate_schedules(df_forecast_dict, c_sample, elite_pool, shifts)
    q_sla, q_asa, q_aband, q_off = evaluate_schedules(df_forecast_dict, q_sample, elite_pool, shifts)
   
    print(f"{'Operational Metric':<25} | {'Classical ILP (PuLP)':<20} | {'Quantum QUBO (neal)':<20}")
    print("-" * 75)
    print(f"{'Execution Time (sec)':<25} | {c_time:<20.4f} | {q_time:<20.4f}")
    print(f"{'Total Agents Deployed':<25} | {c_assigned:<20} | {q_assigned:<20}")
    print(f"{'Projected Payroll':<25} | ${c_cost:<19,.2f} | ${q_cost:<19,.2f}")
    print(f"{'Routing Mismatches':<25} | {c_cross:<20} | {q_cross:<20}")
    print(f"{'Off-Preference Shifts':<25} | {c_off:<20} | {q_off:<20}")
    print(f"{'Service Level (SLA %)':<25} | {c_sla:<19.1f}% | {q_sla:<19.1f}%")
    print(f"{'Avg Speed of Answer':<25} | {c_asa:<18.1f} s | {q_asa:<18.1f} s")
    print(f"{'Queue Abandonment Rate':<25} | {c_aband:<19.2f}% | {q_aband:<19.2f}%")
