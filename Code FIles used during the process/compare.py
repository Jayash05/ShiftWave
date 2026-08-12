import math
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from itertools import combinations
import neal
import pulp

# 1. ADVANCED WFM METRIC FUNCTIONS (Erlang C)
def erlang_c(A, N):
    if N <= A: return 1.0 
    inv_b = 1.0
    for i in range(1, int(N) + 1):
        inv_b = 1.0 + (inv_b * i) / A
    erlang_b = 1.0 / inv_b
    p_w = (N * erlang_b) / (N - A * (1 - erlang_b))
    return min(max(p_w, 0.0), 1.0)

def calculate_erlang_target(hourly_rate, aht, target_asa=20.0, target_sla=0.80):
    if hourly_rate <= 0: return 0
    A = (hourly_rate / 3600.0) * aht
    N = max(1, math.ceil(A))
    while True:
        p_w = erlang_c(A, N)
        service_level = 1.0 - (p_w * math.exp(-(N - A) * (target_asa / aht)))
        if service_level >= target_sla and N > A:
            return N
        N += 1

def evaluate_operational_performance(forecast_df, scheduled_agents_by_interval, aht=240.0):
    total_calls = 0
    total_answered_wait = 0
    total_abandoned = 0
    total_handled = 0
    agent_busy_seconds = 0
    agent_total_capacity_seconds = 0
    
    # [FIX] Added total_sl_calls to properly weight SLA across all intervals
    total_sl_calls = 0  
    
    sl_target_time = 20.0 
    patience_mean = 300.0 
    
    for t_idx, row in forecast_df.iterrows():
        vol = row['Predicted_Volume']
        if vol <= 0: continue
        
        interval_agents = scheduled_agents_by_interval.get(t_idx, 0)
        total_calls += vol
        
        lambda_rate = vol / 900.0 
        A = lambda_rate * aht
        
        if interval_agents > A and interval_agents > 0:
            pw = erlang_c(A, interval_agents)
            asa = (pw * aht) / (interval_agents - A)
            sl = 1.0 - (pw * math.exp(-(interval_agents - A) * (sl_target_time / aht)))
            sl = min(max(sl, 0.0), 1.0)
            
            abandon_rate = pw * math.exp(-patience_mean / max(asa, 1.0))
            abandoned_calls = int(vol * abandon_rate)
            answered_calls = int(vol - abandoned_calls)
            
            total_answered_wait += answered_calls * asa
            total_abandoned += abandoned_calls
            total_handled += answered_calls
            total_sl_calls += answered_calls * sl # to Accumulate SLA successful calls
            
            agent_busy_seconds += answered_calls * aht
            agent_total_capacity_seconds += interval_agents * 900.0
        else:
            abandoned_calls = int(vol * 0.8) 
            total_abandoned += abandoned_calls
            total_handled += (vol - abandoned_calls)
            total_answered_wait += (vol - abandoned_calls) * 450.0

    # Compute weighted SLA instead of terminal-interval SLA
    avg_sla = (total_sl_calls / max(total_handled, 1)) * 100.0 
    avg_asa = total_answered_wait / max(total_handled, 1)
    overall_abandon_rate = (total_abandoned / max(total_calls, 1)) * 100.0
    overall_occupancy = min((agent_busy_seconds / max(agent_total_capacity_seconds, 1)) * 100.0, 100.0)
    
    return avg_sla, avg_asa, overall_abandon_rate, overall_occupancy

# 2. SHARED AI PREDICTION LAYER (FULL SET)
def run_ai_prediction_layer(df_log, df_agents):
    print("\n[1] Running Shared AI Prediction Layer (Full Test Set)...")
    time_col = [c for c in ['Timestamp', 'Arrival_Timestamp'] if c in df_log.columns][0]
    aht_col = [c for c in ['Handle_Time_Seconds', 'Handle_Time_Sec'] if c in df_log.columns][0]
    
    df_log[time_col] = pd.to_datetime(df_log[time_col])
    df_log['Interval_15m'] = df_log[time_col].dt.floor('15min')
    
    agg = df_log.groupby('Interval_15m').agg(
        Call_Count=('Call_ID', 'count'), Avg_AHT=(aht_col, 'mean')
    ).reset_index().sort_values('Interval_15m').reset_index(drop=True)
    
    agg['Hour'] = agg['Interval_15m'].dt.hour
    agg['Minute'] = agg['Interval_15m'].dt.minute
    agg['DayOfWeek'] = agg['Interval_15m'].dt.dayofweek
    
    X = agg[['Hour', 'Minute', 'DayOfWeek']]
    y_vol = agg['Call_Count']
    
    split_idx = int(len(agg) * 0.8)
    vol_model = RandomForestRegressor(n_estimators=30, random_state=42)
    vol_model.fit(X.iloc[:split_idx], y_vol.iloc[:split_idx])
    
    future_forecast = agg.iloc[split_idx:].copy().reset_index(drop=True)
    future_forecast['Predicted_Volume'] = np.ceil(vol_model.predict(X.iloc[split_idx:]) * 1.05)
    future_forecast['Predicted_AHT'] = 240.0
    
    future_forecast['Target_Headcount'] = future_forecast.apply(
        lambda r: math.ceil(calculate_erlang_target(r['Predicted_Volume'] * 4, r['Predicted_AHT']) * 1.15), axis=1
    )
    
    # [FIX 2] Scale agent roster dynamically based on test set duration
    peak_demand = future_forecast['Target_Headcount'].max()
    days_in_test = max(1.0, len(future_forecast) / 96.0)
    shifts_needed = 3 * days_in_test
    agents_needed = int(peak_demand * shifts_needed * 1.3)
    
    elite_agents = df_agents.head(min(agents_needed, len(df_agents))).reset_index(drop=True)
    
    print(f"Forecast Horizon: {days_in_test:.1f} Days")
    print(f" Total Volume Predicted: {int(future_forecast['Predicted_Volume'].sum())} calls")
    print(f"Scaled Roster Capacity: Provided {len(elite_agents)} agents for full multi-day coverage.")
    return future_forecast, elite_agents

# 3. MULTI-DAY SOLVERS
def run_classical_baseline(df_forecast, df_agents, shifts):
    print("\n[2] Executing Classical Linear Programming (PuLP) on Full Horizon...")
    start_time = time.time()
    prob = pulp.LpProblem("Schedule_Optimization", pulp.LpMinimize)
    agent_vars = {}
    
    for i in df_agents.index:
        for s in shifts.keys():
            agent_vars[(i, s)] = pulp.LpVariable(f"x_{i}_{s}", cat='Binary')
            
    objective = []
    days_in_test = len(df_forecast) / 96.0
    for i, agent in df_agents.iterrows():
        pref = str(agent.get('Preferred_Shift', 'Morning'))
        cost_ph = float(agent.get('Hourly_Wage', 25.0))
        for s in shifts.keys():
            if s == 'Off': continue
            penalty = 0 if s in pref else 50 
            objective.append(agent_vars[(i, s)] * ((cost_ph * 8 * days_in_test) + penalty))
            
    prob += pulp.lpSum(objective)
    
    for i in df_agents.index:
        prob += pulp.lpSum([agent_vars[(i, s)] for s in shifts.keys()]) == 1
        
    for t_idx, row in df_forecast.iterrows():
        target = row['Target_Headcount']
        if target <= 0: continue
        active_vars = [agent_vars[(i, s)] for i in df_agents.index for s, s_arr in shifts.items() if s_arr[t_idx] == 1]
        if active_vars:
            prob += pulp.lpSum(active_vars) >= target
        
    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=60))
    duration = time.time() - start_time
    
    assigned, off_pref, cost = 0, 0, 0.0
    interval_counts = {t_idx: 0 for t_idx in range(len(df_forecast))}
    
    for i, agent in df_agents.iterrows():
        pref = str(agent.get('Preferred_Shift', 'Morning'))
        for s, s_arr in shifts.items():
            if s != 'Off' and pulp.value(agent_vars[(i, s)]) == 1:
                assigned += 1
                cost += float(agent.get('Hourly_Wage', 25.0)) * 8 * days_in_test
                if s not in pref: off_pref += 1
                for t_idx, active in enumerate(s_arr):
                    if active: interval_counts[t_idx] += 1
                
    sla, asa, abandon, occupancy = evaluate_operational_performance(df_forecast, interval_counts)
    return duration, assigned, off_pref, cost, sla, asa, abandon, occupancy

def run_quantum_qubo(df_forecast, df_agents, shifts):
    print("\n[3] Executing Native QUBO on D-Wave Simulated Annealer (Full Horizon)...")
    start_time = time.time()
    Q = {}
    L1_DEMAND, L2_SINGLE, L3_FRICTION = 100.0, 5000.0, 50.0
    num_agents, num_shifts = len(df_agents), len(shifts)
    days_in_test = len(df_forecast) / 96.0
    
    for i in range(num_agents):
        for s1 in range(num_shifts):
            idx1 = i * num_shifts + s1
            Q[(idx1, idx1)] = Q.get((idx1, idx1), 0) - L2_SINGLE
            for s2 in range(s1 + 1, num_shifts):
                idx2 = i * num_shifts + s2
                Q[(idx1, idx2)] = Q.get((idx1, idx2), 0) + (2 * L2_SINGLE)
                
    for i, agent in df_agents.iterrows():
        pref = str(agent.get('Preferred_Shift', 'Morning'))
        cost_ph = float(agent.get('Hourly_Wage', 25.0))
        for s_idx, (s_name, _) in enumerate(shifts.items()):
            if s_name == 'Off': continue
            idx = i * num_shifts + s_idx
            penalty = 0 if s_name in pref else L3_FRICTION
            Q[(idx, idx)] = Q.get((idx, idx), 0) + penalty + (cost_ph * 8 * days_in_test)

    slack_offset = num_agents * num_shifts
    curr_slack = slack_offset
    
    for t_idx, row in df_forecast.iterrows():
        target = row['Target_Headcount']
        if target <= 0: continue
        
        active_qubits = [i * num_shifts + s_idx for i in range(num_agents) for s_idx, (s_name, s_arr) in enumerate(shifts.items()) if s_arr[t_idx] == 1]
        max_slack = max(1, math.ceil(math.log2(target + 5)))
        slack_vars = [(curr_slack + k, 2**k) for k in range(max_slack)]
        curr_slack += max_slack
            
        all_vars = [(q, 1) for q in active_qubits] + [(sq, -sw) for sq, sw in slack_vars]
        for v1, w1 in all_vars:
            Q[(v1, v1)] = Q.get((v1, v1), 0) + L1_DEMAND * (w1**2 - 2 * target * w1)
            for v2, w2 in all_vars:
                if v1 < v2: Q[(v1, v2)] = Q.get((v1, v2), 0) + (2 * L1_DEMAND * w1 * w2)

    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q, num_reads=50)
    best = response.first.sample
    duration = time.time() - start_time
    
    assigned, off_pref, cost = 0, 0, 0.0
    interval_counts = {t_idx: 0 for t_idx in range(len(df_forecast))}
    
    for i, agent in df_agents.iterrows():
        pref = str(agent.get('Preferred_Shift', 'Morning'))
        for s_idx, (s_name, s_arr) in enumerate(shifts.items()):
            idx = i * num_shifts + s_idx
            if best.get(idx, 0) == 1 and s_name != 'Off':
                assigned += 1
                cost += float(agent.get('Hourly_Wage', 25.0)) * 8 * days_in_test
                if s_name not in pref: off_pref += 1
                for t_idx, active in enumerate(s_arr):
                    if active: interval_counts[t_idx] += 1
                
    sla, asa, abandon, occupancy = evaluate_operational_performance(df_forecast, interval_counts)
    return duration, assigned, off_pref, cost, sla, asa, abandon, occupancy

# MASTER BENCHMARK EXECUTION
if __name__ == "__main__":    
    try:
        df_log = pd.read_csv("call_log.csv")
        df_agents = pd.read_csv("agent_data.csv")
    except Exception as e:
        print(f"Error loading files: {e}")
        exit()
        
    df_forecast, elite_agents = run_ai_prediction_layer(df_log, df_agents)
    
    # MULTI-DAY SHIFT MATRIX UPGRADE
    total_intervals = len(df_forecast)
    shifts = {
        'Morning': [1 if 28 <= (t % 96) < 64 else 0 for t in range(total_intervals)],
        'Midday':  [1 if 40 <= (t % 96) < 76 else 0 for t in range(total_intervals)],
        'Evening': [1 if 52 <= (t % 96) < 88 else 0 for t in range(total_intervals)],
        'Off':     [0 for _ in range(total_intervals)]
    }
    
    c_time, c_assign, c_off, c_cost, c_sla, c_asa, c_aband, c_occ = run_classical_baseline(df_forecast, elite_agents, shifts)
    q_time, q_assign, q_off, q_cost, q_sla, q_asa, q_aband, q_occ = run_quantum_qubo(df_forecast, elite_agents, shifts)
    
    print(" ULTIMATE HEAD-TO-HEAD ENTERPRISE SCORECARD")
    print(f"{'Operational Metric':<25} | {'Classical (PuLP)':<18} | {'Hybrid Quantum':<18}")
    print("-" * 68)
    print(f"{'Execution Time (sec)':<25} | {c_time:>18.4f} | {q_time:>18.4f}")
    print(f"{'Total Agents Assigned':<25} | {c_assign:>18} | {q_assign:>18}")
    print(f"{'Total Horizon Payroll':<25} | ${c_cost:>16,.2f} | ${q_cost:>16,.2f}")
    print(f"{'Off-Preference Count':<25} | {c_off:>18} | {q_off:>18}")
    print(f"{'Service Level (SLA %)':<25} | {c_sla:>17.1f}% | {q_sla:>17.1f}%")
    print(f"{'Avg Speed of Answer':<25} | {c_asa:>15.1f}s | {q_asa:>15.1f}s")
    print(f"{'Queue Abandonment Rate':<25} | {c_aband:>17.2f}% | {q_aband:>17.2f}%")
    print(f"{'Agent Occupancy Rate':<25} | {c_occ:>17.1f}% | {q_occ:>17.1f}%")
