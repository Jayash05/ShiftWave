import math
import time
import os
import pandas as pd
import dimod
import neal

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

# 2. EVALUATION ENGINE A: ERLANG C (MACRO)
def evaluate_schedules(df_forecast_dict, best_sample, df_agents, shifts):
    schedule_counts = {q: {t: 0 for t in range(96)} for q in df_forecast_dict.keys()}
    off_pref = 0
    
    for var_name, val in best_sample.items():
        if val == 1 and var_name.startswith('x_') and not var_name.endswith('_None'):
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

# 3. EVALUATION: EVENT SIMULATOR (MICRO)
def simulate_agent_workday(df_calls, best_sample, shifts):
    print("\n[DIAGNOSTIC] Running Discrete Event Simulation on Quantum Schedule...")
    
    scheduled_agents = {}
    for var, val in best_sample.items():
        if val == 1 and var.startswith('x_') and not var.endswith('_None'):
            parts = var.split('_')
            a_id = f"{parts[1]}_{parts[2]}"
            s_name = parts[3]
            q_name = "_".join(parts[4:])
            
            shift_intervals = [i for i, v in enumerate(shifts[s_name]) if v == 1]
            if not shift_intervals: continue
            
            start_sec = shift_intervals[0] * 900
            end_sec = (shift_intervals[-1] + 1) * 900
            
            scheduled_agents[a_id] = {
                'Queue': q_name,
                'Shift': s_name,
                'Start_Sec': start_sec,
                'End_Sec': end_sec,
                'Next_Available': start_sec, 
                'Calls_Taken': 0,
                'Idle_Gaps_Sec': [] 
            }
            
    day_calls = df_calls.sort_values('Arrival_Timestamp')
    midnight = day_calls['Arrival_Timestamp'].dt.normalize().iloc[0]
    
    for _, call in day_calls.iterrows():
        call_sec = (call['Arrival_Timestamp'] - midnight).total_seconds()
        q_name = call['Queue']
        aht = call['Handle_Time_Seconds']
        
        available_agents = []
        for a_id, data in scheduled_agents.items():
            if data['Queue'] == q_name and data['Start_Sec'] <= call_sec < data['End_Sec']:
                if data['Next_Available'] <= call_sec:
                    available_agents.append(a_id)
                    
        if available_agents:
            chosen_agent = min(available_agents, key=lambda x: scheduled_agents[x]['Next_Available'])
            idle_time = call_sec - scheduled_agents[chosen_agent]['Next_Available']
            scheduled_agents[chosen_agent]['Idle_Gaps_Sec'].append(idle_time)
            
            scheduled_agents[chosen_agent]['Next_Available'] = call_sec + aht
            scheduled_agents[chosen_agent]['Calls_Taken'] += 1

    results = []
    for a_id, data in scheduled_agents.items():
        avg_idle_mins = (sum(data['Idle_Gaps_Sec']) / len(data['Idle_Gaps_Sec']) / 60) if data['Idle_Gaps_Sec'] else 0
        results.append({
            'Agent_ID': a_id,
            'Shift': data['Shift'],
            'Calls_Taken': data['Calls_Taken'],
            'Avg_Idle_Between_Calls_Mins': avg_idle_mins
        })
        
    return pd.DataFrame(results)

# 4. MODULE A: CLASSICAL MACHINE
def calculate_classical_shift_targets(df_forecast_dict, shifts):
    print("\n[MODULE A] Classical Phase: Calculating Temporal SLA Bounds...")
    start_time = time.time()
    
    shift_targets = {}
    shift_names = [s for s in shifts.keys() if s != 'Off']
    
    for q_name, df_forecast in df_forecast_dict.items():
        shift_targets[q_name] = {s: 0 for s in shift_names}
        for s_name in shift_names:
            max_needed = 0
            for t_idx, row in df_forecast.iterrows():
                if shifts[s_name][t_idx] == 1:
                    if row['Target_Headcount'] > max_needed:
                        max_needed = row['Target_Headcount']
            
            shift_targets[q_name][s_name] = int(math.ceil(max_needed * 1.02))
            
    exec_time = time.time() - start_time
    return shift_targets, exec_time

# 5. MODULE B: HIGH-RESOLUTION BQM ASSIGNMENT
def execute_high_res_quantum(df_forecast_dict, df_agents, shifts, queue_metadata):
    total_start = time.time()
    shift_targets, _ = calculate_classical_shift_targets(df_forecast_dict, shifts)
    
    print("[MODULE B] Processing Trimmed Agent Pool (Micro-Clustering)...")
    squads = []
    
    SQUAD_SIZE = 5 
    grouped = df_agents.groupby(['Tier', 'Preferred_Shift'])
    
    for (tier, pref), group in grouped:
        agents = group.to_dict('records')
        for i in range(0, len(agents), SQUAD_SIZE):
            chunk = agents[i:i + SQUAD_SIZE]
            
            squad_id = f"SQ_{tier}_{pref[:3]}_{i//SQUAD_SIZE}"
            squad_wage = sum(float(a['Hourly_Wage']) * float(a.get('Shift_Elapsed_Hours', 8.0)) for a in chunk)
            squad_m_i = sum(float(a['Tenure_AHT_Multiplier']) * float(a['Micro_Break_Capacity_Multiplier']) for a in chunk)
            
            rep_agent = chunk[0]
            valid_queues = [q for q, meta in queue_metadata.items() if rep_agent.get(meta['skill_col'], 0) == 1]
            
            squads.append({
                'squad_id': squad_id,
                'agents': chunk, 
                'cost': squad_wage,
                'm_i': int(round(squad_m_i)), 
                'pref': str(pref),
                'tier': {'General': 1, 'Advisor': 2, 'Intermed': 3, 'Investor': 4}.get(tier, 2),
                'valid_queues': valid_queues
            })

    print("[MODULE B] Building Matrix and Expanding Algebra...")
    shift_names = [s for s in shifts.keys() if s != 'Off']
    linear = {}
    quadratic = {}
    
    def add_linear(v, weight):
        linear[v] = linear.get(v, 0.0) + weight
        
    def add_quad(v1, v2, weight):
        if v1 > v2: v1, v2 = v2, v1
        quadratic[(v1, v2)] = quadratic.get((v1, v2), 0.0) + weight

    PENALTY_1 = 50000.0
    for sq in squads:
        sq_vars = []
        for s_name in shift_names:
            for q_name in sq['valid_queues']:
                var_name = f"x_{sq['squad_id']}_{s_name}_{q_name}"
                sq_vars.append(var_name)
                
                q_tier = queue_metadata[q_name]['tier']
                friction = 0 if s_name in sq['pref'] else 100.0 * len(sq['agents'])
                routing = 200.0 * ((sq['tier'] - q_tier)**2)
                add_linear(var_name, (sq['cost'] + friction + routing) * 0.002)
                
        off_var = f"x_{sq['squad_id']}_Off_None"
        all_vars = sq_vars + [off_var]
        add_linear(off_var, 0.0)
        
        for var in all_vars: add_linear(var, -PENALTY_1)
        for i in range(len(all_vars)):
            for j in range(i+1, len(all_vars)):
                add_quad(all_vars[i], all_vars[j], 2 * PENALTY_1)

    # MASSIVE SLA ENFORCER to reduce the Abandonment rate
    PENALTY_2 = 250.0 
    slot_vars = {q: {s: [] for s in shift_names} for q in queue_metadata.keys()}
    
    for sq in squads:
        for s_name in shift_names:
            for q_name in sq['valid_queues']:
                var_name = f"x_{sq['squad_id']}_{s_name}_{q_name}"
                slot_vars[q_name][s_name].append((var_name, sq['m_i']))

    for q_name, s_dict in slot_vars.items():
        for s_name, active_sqs in s_dict.items():
            target_N = shift_targets.get(q_name, {}).get(s_name, 0)
            if target_N == 0: 
                for var_name, _ in active_sqs: add_linear(var_name, PENALTY_2 * 100)
                continue
                
            shifted_target = target_N + 5
            
            for var_name, c_i in active_sqs:
                weight = PENALTY_2 * ((c_i ** 2) - 2 * shifted_target * c_i)
                add_linear(var_name, weight)
            
            for i in range(len(active_sqs)):
                for j in range(i+1, len(active_sqs)):
                    v1, c1 = active_sqs[i]
                    v2, c2 = active_sqs[j]
                    add_quad(v1, v2, PENALTY_2 * 2 * c1 * c2)

    print("[MODULE B] Running D-Wave Simulated Annealer (Balanced for Speed/Res)...")
    bqm = dimod.BinaryQuadraticModel(linear, quadratic, 0.0, 'BINARY')
    sampler = neal.SimulatedAnnealingSampler()
    
    sampleset = sampler.sample(bqm, num_reads=25, num_sweeps=300)
    best_sample = sampleset.first.sample
    
    total_time = time.time() - total_start
    
    print("[PIPELINE COMPLETE] Unpacking Agents...")
    final_dict = {}
    assigned, cost, cross_skill = 0, 0.0, 0
    assigned_agents = set()
    
    for sq in squads:
        squad_assigned = False
        for s_name in shift_names:
            for q_name in sq['valid_queues']:
                var_name = f"x_{sq['squad_id']}_{s_name}_{q_name}"
                if best_sample.get(var_name, 0.0) > 0.5 and not squad_assigned:
                    squad_assigned = True
                    q_tier = queue_metadata[q_name]['tier']
                    
                    for agent in sq['agents']:
                        if agent['Agent_ID'] not in assigned_agents:
                            assigned_agents.add(agent['Agent_ID'])
                            indiv_var = f"x_{agent['Agent_ID']}_{s_name}_{q_name}"
                            final_dict[indiv_var] = 1
                            assigned += 1
                            cost += float(agent['Hourly_Wage']) * float(agent.get('Shift_Elapsed_Hours', 8.0))
                            if sq['tier'] != q_tier: cross_skill += 1
                else:
                    for agent in sq['agents']:
                        indiv_var = f"x_{agent['Agent_ID']}_{s_name}_{q_name}"
                        if indiv_var not in final_dict: final_dict[indiv_var] = 0

    return final_dict, total_time, assigned, cost, cross_skill

if __name__ == "__main__":
    print("[1] Checking for datasets...")
    
    if not os.path.exists("call_log.csv") or not os.path.exists("agent_data.csv"):
        print("CSV files not found in Colab! Please upload them using the prompt below:")
        from google.colab import files
        uploaded = files.upload()
        
    try:
        df_calls = pd.read_csv("call_log.csv")
        df_agents = pd.read_csv("agent_data.csv")
    except FileNotFoundError:
        print("FATAL ERROR: Files still not found. Please upload them.")
        raise SystemExit("Stopping execution.")
        
    print("[2] Processing Call Data & Generating Erlang C Targets for MONDAY...")
    df_calls['Arrival_Timestamp'] = pd.to_datetime(df_calls['Arrival_Timestamp'])
    day_calls = df_calls[(df_calls['Day_of_Week'] == 'Monday') & (df_calls['Handled_By'] == 'Human')].copy()
    
    queue_metadata = {
        'General_Support': {'skill_col': 'Skill_GenSupport', 'tier': 1},
        'Advisor':         {'skill_col': 'Skill_Advisor',    'tier': 2},
        'Intermed':        {'skill_col': 'Skill_Intermed',   'tier': 3},
        'Investor':        {'skill_col': 'Skill_Investor',   'tier': 4}
    }
    
    df_forecast_dict = {}
    for q in day_calls['Queue'].unique():
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

    shifts = {
        'Morning': [1 if 24 <= t < 56 else 0 for t in range(96)],
        'Midday':  [1 if 40 <= t < 72 else 0 for t in range(96)],
        'Evening': [1 if 56 <= t < 88 else 0 for t in range(96)],
        'Off':     [0 for _ in range(96)]
    }
    
    # DYNAMIC POOL TRIMMING
    total_peak_needed = sum(interval_stats['Target_Headcount'].max() for interval_stats in df_forecast_dict.values())
    pool_size = int(total_peak_needed * 1.5)
    
    if 'Contract_Type' in df_agents.columns:
        df_agents['Is_FT'] = (df_agents['Contract_Type'] == 'Full-Time').astype(int)
        df_agents = df_agents.sort_values(by=['Is_FT', 'Tenure_AHT_Multiplier'], ascending=[False, False])
        
    df_agents_trimmed = df_agents.head(pool_size).copy()
    
    q_sample, q_time, q_assigned, q_cost, q_cross = execute_high_res_quantum(df_forecast_dict, df_agents_trimmed, shifts, queue_metadata)
    
    # RUN EVALUATOR  (ERLANG C)
    q_sla, q_asa, q_aband, q_off = evaluate_schedules(df_forecast_dict, q_sample, df_agents_trimmed, shifts)
    
    # RUN EVALUATOR 1 (EVENT SIMULATOR)
    df_agent_stats = simulate_agent_workday(day_calls, q_sample, shifts)
    
    active_agents = df_agent_stats[df_agent_stats['Calls_Taken'] > 0]
    avg_idle_time = active_agents['Avg_Idle_Between_Calls_Mins'].mean() if not active_agents.empty else 0
    zero_call_agents = len(df_agent_stats[df_agent_stats['Calls_Taken'] == 0])
    busiest_agent_calls = active_agents['Calls_Taken'].max() if not active_agents.empty else 0
    
   
    print("--- MACRO METRICS (ERLANG C) ---")
    print(f"{'Execution Time (sec)':<30} | {q_time:<20.4f}")
    print(f"{'Total Agents Deployed':<30} | {q_assigned:<20}")
    print(f"{'Projected Payroll':<30} | ${q_cost:<19,.2f}")
    print(f"{'Service Level (SLA %)':<30} | {q_sla:<19.1f}%")
    print(f"{'Avg Speed of Answer':<30} | {q_asa:<18.1f} s")
    print(f"{'Queue Abandonment Rate':<30} | {q_aband:<19.2f}%")
    
    print("\n--- MICRO METRICS (EVENT SIMULATOR) ---")
    print(f"{'Avg Idle Time Between Calls':<30} | {avg_idle_time:<18.1f} mins")
    print(f"{'Agents Who Took ZERO Calls':<30} | {zero_call_agents:<20}")
    print(f"{'Max Calls Taken by One Agent':<30} | {busiest_agent_calls:<20}")
    
    if not active_agents.empty:
        print("\n🚨 EXTREME OVERWORK (Top 3 Busiest Agents):")
        print(active_agents.sort_values('Calls_Taken', ascending=False).head(3).to_string(index=False))
        
        print("\n💤 EXTREME IDLE TIME (Top 3 Least Busy Agents):")
        print(active_agents.sort_values('Calls_Taken', ascending=True).head(3).to_string(index=False))
