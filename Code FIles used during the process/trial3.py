"""
Vanguard 4.1: THE HACKATHON WINNER
Classical vs. Quantum-Inspired Local Simulator (Custom Schema Match)
"""

import math
import time
import pandas as pd
import pulp
import dimod
import neal

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

# ==========================================
# 2. EVALUATION ENGINE
# ==========================================
def evaluate_schedules(df_forecast_dict, best_sample, df_agents, shifts):
    schedule_counts = {q: {t: 0 for t in range(96)} for q in df_forecast_dict.keys()}
    off_pref = 0
    
    for var_name, val in best_sample.items():
        if val == 1 and var_name.startswith('x_'):
            parts = var_name.split('_')
            a_id = f"{parts[1]}_{parts[2]}"
            s_name = parts[3]
            q_name = "_".join(parts[4:])
            
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

# ==========================================
# 3. CLASSICAL LP BASELINE (PuLP)
# ==========================================
def execute_classical_model(df_forecast_dict, df_agents, shifts, queue_metadata):
    print("\n[CLASSICAL] Executing PuLP Linear Programming...")
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
    # Give classical 60s max to keep things competitive
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
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
# ==========================================
# 4. TRUE QUANTUM SIMULATOR (MACRO-QUBITS)
# ==========================================
def execute_true_quantum_squads(df_forecast_dict, df_agents, shifts, queue_metadata):
    print("\n[QUANTUM] Clustering Agents into Macro-Qubits (Squads)...")
    start_time = time.time()
    
    # 1. Cluster identical agents to shrink the Hamiltonian
    squads = []
    # Group by Tier and Preferred Shift
    grouped = df_agents.groupby(['Tier', 'Preferred_Shift'])
    
    SQUAD_SIZE = 12 # Compresses 3500 agents into ~290 Qubits
    
    for (tier, pref), group in grouped:
        agents = group.to_dict('records')
        for i in range(0, len(agents), SQUAD_SIZE):
            chunk = agents[i:i + SQUAD_SIZE]
            
            # Aggregate squad stats
            squad_id = f"SQ_{tier}_{pref[:3]}_{i//SQUAD_SIZE}"
            squad_wage = sum(float(a['Hourly_Wage']) * float(a.get('Shift_Elapsed_Hours', 8.0)) for a in chunk)
            squad_m_i = sum(float(a['Tenure_AHT_Multiplier']) * float(a['Micro_Break_Capacity_Multiplier']) for a in chunk)
            
            # Assume all agents in this chunk share the exact same valid queues (based on tier)
            rep_agent = chunk[0]
            valid_queues = [q for q, meta in queue_metadata.items() if rep_agent.get(meta['skill_col'], 0) == 1]
            
            squads.append({
                'squad_id': squad_id,
                'agents': chunk,
                'cost': squad_wage,
                'm_i': squad_m_i,
                'pref': str(pref),
                'tier': {'General': 1, 'Advisor': 2, 'Intermed': 3, 'Investor': 4}.get(tier, 2),
                'valid_queues': valid_queues
            })

    print(f"    -> Compressed {len(df_agents)} agents into {len(squads)} manageable Qubits.")
    
    # 2. Build the Constrained Quadratic Model
    print("[QUANTUM] Compiling True Constrained Quadratic Model (CQM)...")
    cqm = dimod.ConstrainedQuadraticModel()
    x_vars = {}
    shift_names = [s for s in shifts.keys() if s != 'Off']
    
    for sq in squads:
        for s_name in shift_names:
            for q_name in sq['valid_queues']:
                var_name = f"x_{sq['squad_id']}_{s_name}_{q_name}"
                x_vars[(sq['squad_id'], s_name, q_name)] = dimod.Binary(var_name)
                
    # Hard Constraint: Max one shift per Squad
    for sq in squads:
        sq_vars = [x_vars[(sq['squad_id'], s_name, q_name)] 
                   for s_name in shift_names for q_name in sq['valid_queues']]
        if sq_vars:
            cqm.add_constraint(sum(sq_vars) <= 1, label=f"MaxOneShift_{sq['squad_id']}")
            
    # Hard Constraint: Demand Coverage (Integer Scaled for RAM Safety)
    for q_name, df_forecast in df_forecast_dict.items():
        meta = queue_metadata[q_name]
        q_tier = meta['tier']
        for t_idx, row in df_forecast.iterrows():
            target_N = row['Target_Headcount']
            if target_N <= 0: continue
            
            target_N_scaled = int(target_N * 10)
            active_vars = []
            
            for sq in squads:
                if q_name in sq['valid_queues']:
                    m_i_scaled = int(round(sq['m_i'] * 10))
                    for s_name in shift_names:
                        if shifts[s_name][t_idx] == 1:
                            active_vars.append(m_i_scaled * x_vars[(sq['squad_id'], s_name, q_name)])
                            
            if active_vars:
                cqm.add_constraint(sum(active_vars) >= target_N_scaled, label=f"Demand_{q_name}_{t_idx}")
                
    # Objective Formulation
    L3_FRICTION, L4_ROUTING = 500.0, 1500.0
    SCALING_FACTOR = 0.005
    objective_terms = []
    
    for sq in squads:
        for s_name in shift_names:
            for q_name in sq['valid_queues']:
                q_tier = queue_metadata[q_name]['tier']
                
                h_cost = sq['cost']
                h_friction = 0 if s_name in sq['pref'] else L3_FRICTION * len(sq['agents'])
                h_routing = L4_ROUTING * ((sq['tier'] - q_tier)**2) * sum(shifts[s_name])
                
                total_weight = (h_cost + h_friction + h_routing) * SCALING_FACTOR
                objective_terms.append(total_weight * x_vars[(sq['squad_id'], s_name, q_name)])
                
    cqm.set_objective(sum(objective_terms))
    
    # 3. Flatten and Execute using D-Wave Neal
    print("[QUANTUM] Flattening CQM to Binary Quadratic Model (BQM)...")
    bqm, invert = dimod.cqm_to_bqm(cqm, lagrange_multiplier=5.0)
    
    print("[QUANTUM] Running D-Wave Simulated Annealer (neal)...")
    sampler = neal.SimulatedAnnealingSampler()
    sampleset = sampler.sample(bqm, num_reads=25, num_sweeps=500)
    
    best_bqm_sample = sampleset.first.sample
    best_sample = invert(best_bqm_sample)
    
    duration = time.time() - start_time
    
    # 4. Unpack Squads back into individual Agents for the Evaluation Engine
    print("[QUANTUM] Unpacking Macro-Qubits to Individual Agents...")
    final_dict = {}
    assigned, cost, cross_skill = 0, 0.0, 0
    
    for sq in squads:
        for s_name in shift_names:
            for q_name in sq['valid_queues']:
                var_name = f"x_{sq['squad_id']}_{s_name}_{q_name}"
                val = best_sample.get(var_name, 0.0)
                
                if val > 0.5:
                    q_tier = queue_metadata[q_name]['tier']
                    # Reconstruct the individual agent variables so the Evaluate function can read it
                    for agent in sq['agents']:
                        indiv_var = f"x_{agent['Agent_ID']}_{s_name}_{q_name}"
                        final_dict[indiv_var] = 1
                        assigned += 1
                        cost += float(agent['Hourly_Wage']) * float(agent.get('Shift_Elapsed_Hours', 8.0))
                        if sq['tier'] != q_tier: cross_skill += 1
                else:
                    for agent in sq['agents']:
                        indiv_var = f"x_{agent['Agent_ID']}_{s_name}_{q_name}"
                        final_dict[indiv_var] = 0

    return final_dict, duration, assigned, cost, cross_skill
# ==========================================
# 5. MASTER EXECUTION
# ==========================================
if __name__ == "__main__":
    print("[1] Loading existing datasets (agent_data.csv & call_log.csv)...")
    try:
        df_calls = pd.read_csv("call_log.csv")
        df_agents = pd.read_csv("agent_data.csv")
    except FileNotFoundError:
        print("FATAL ERROR: CSV files not found.")
        exit()
        
    print("[2] Processing Call Data & Generating Erlang C Targets...")
    df_calls['Arrival_Timestamp'] = pd.to_datetime(df_calls['Arrival_Timestamp'])
    target_date = df_calls['Arrival_Timestamp'].dt.date.min()
    # Ensure we only calculate humans
    day_calls = df_calls[(df_calls['Arrival_Timestamp'].dt.date == target_date) & (df_calls['Handled_By'] == 'Human')].copy()
    
    # NOTE: Updated to exactly match your dataset columns (Skill_GenSupport)
    # ---------------- REPLACEMENT START ----------------
    queue_metadata = {
        'General_Support': {'skill_col': 'Skill_GenSupport', 'tier': 1},
        'Advisor':         {'skill_col': 'Skill_Advisor',    'tier': 2},
        'Intermed':        {'skill_col': 'Skill_Intermed',   'tier': 3},
        'Trust_Intermed':  {'skill_col': 'Skill_Intermed',   'tier': 3}, # Explicitly added
        'Investor':        {'skill_col': 'Skill_Investor',   'tier': 4}
    }
    
    df_forecast_dict = {}
    for q in day_calls['Queue'].unique():
        # BULLETPROOF SAFEGUARD: If a new queue appears, map it dynamically
        if q not in queue_metadata:
            if 'Intermed' in q:   skill, tier = 'Skill_Intermed', 3
            elif 'Advisor' in q:  skill, tier = 'Skill_Advisor', 2
            elif 'Investor' in q: skill, tier = 'Skill_Investor', 4
            else:                 skill, tier = 'Skill_GenSupport', 1
            queue_metadata[q] = {'skill_col': skill, 'tier': tier}
            
        q_calls = day_calls[day_calls['Queue'] == q].copy()
        q_calls.set_index('Arrival_Timestamp', inplace=True)
        interval_stats = q_calls.resample('15Min').agg(Volume=('Call_ID', 'count'), AHT=('Handle_Time_Seconds', 'mean')).fillna(0)
        interval_stats['Target_Headcount'] = interval_stats.apply(
            lambda row: calculate_erlang_target(row['Volume'] * 4, row['AHT'] if row['AHT'] > 0 else 240.0), axis=1)
        df_forecast_dict[q] = interval_stats.reset_index()
    # ---------------- REPLACEMENT END ----------------

    shifts = {
        'Morning': [1 if 24 <= t < 56 else 0 for t in range(96)],
        'Midday':  [1 if 40 <= t < 72 else 0 for t in range(96)],
        'Evening': [1 if 56 <= t < 88 else 0 for t in range(96)],
        'Off':     [0 for _ in range(96)]
    }
    
    # Run Solvers
    c_sample, c_time, c_assigned, c_cost, c_cross = execute_classical_model(df_forecast_dict, df_agents, shifts, queue_metadata)
    q_sample, q_time, q_assigned, q_cost, q_cross = execute_true_quantum_squads(df_forecast_dict, df_agents, shifts, queue_metadata)
    
    # Evaluate Outcomes
    c_sla, c_asa, c_aband, c_off = evaluate_schedules(df_forecast_dict, c_sample, df_agents, shifts)
    q_sla, q_asa, q_aband, q_off = evaluate_schedules(df_forecast_dict, q_sample, df_agents, shifts)
    
    print("\n" + "="*75)
    print(" 📊 VANGUARD ENTERPRISE SCORECARD (CLASSICAL VS. QUANTUM-INSPIRED)")
    print("="*75)
    print(f"{'Operational Metric':<25} | {'Classical ILP (PuLP)':<20} | {'Quantum-Inspired':<20}")
    print("-" * 75)
    print(f"{'Execution Time (sec)':<25} | {c_time:<20.4f} | {q_time:<20.4f}")
    print(f"{'Total Agents Deployed':<25} | {c_assigned:<20} | {q_assigned:<20}")
    print(f"{'Projected Payroll':<25} | ${c_cost:<19,.2f} | ${q_cost:<19,.2f}")
    print(f"{'Routing Mismatches':<25} | {c_cross:<20} | {q_cross:<20}")
    print(f"{'Off-Preference Shifts':<25} | {c_off:<20} | {q_off:<20}")
    print(f"{'Service Level (SLA %)':<25} | {c_sla:<19.1f}% | {q_sla:<19.1f}%")
    print(f"{'Avg Speed of Answer':<25} | {c_asa:<18.1f} s | {q_asa:<18.1f} s")
    print(f"{'Queue Abandonment Rate':<25} | {c_aband:<19.2f}% | {q_aband:<19.2f}%")
    print("="*75)