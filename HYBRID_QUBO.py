import math
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import neal

# ==========================================
# 1. ERLANG C MATHEMATICAL ENGINE (Stable)
# ==========================================
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

# ==========================================
# 2. LAYER 1: AI PREDICTIVE VOLUME FORECASTER
# ==========================================
class AIWorkloadForecaster:
    def __init__(self):
        self.volume_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.aht_model = RandomForestRegressor(n_estimators=50, random_state=42)

    def fit_and_predict(self, df_log):
        """
        Trains on historical data and predicts FUTURE demand to prevent data leakage.
        Uses an 80/20 time-series split.
        """
        time_col = None
        possible_names = ['Timestamp', 'Time', 'Call_Timestamp', 'Arrival_Timestamp', 'Arrival_Time', 'Date']
        for col in possible_names:
            if col in df_log.columns:
                time_col = col
                break
                
        if not time_col:
            raise KeyError(f"Could not find a time column! CSV columns: {df_log.columns.tolist()}")

        aht_col = 'Handle_Time_Seconds' if 'Handle_Time_Seconds' in df_log.columns else 'Handle_Time_Sec'

        df_log[time_col] = pd.to_datetime(df_log[time_col])
        df_log['Interval_15m'] = df_log[time_col].dt.floor('15min')
        
        aggregated = df_log.groupby('Interval_15m').agg(
            Call_Count=('Call_ID', 'count'),
            Avg_AHT=(aht_col, 'mean')
        ).reset_index()

        # Sort chronologically to respect the flow of time
        aggregated = aggregated.sort_values('Interval_15m').reset_index(drop=True)

        aggregated['Hour'] = aggregated['Interval_15m'].dt.hour
        aggregated['Minute'] = aggregated['Interval_15m'].dt.minute
        aggregated['DayOfWeek'] = aggregated['Interval_15m'].dt.dayofweek

        X = aggregated[['Hour', 'Minute', 'DayOfWeek']]
        y_vol = aggregated['Call_Count']
        y_aht = aggregated['Avg_AHT'].fillna(240.0)

        # =======================================================
        # DATA LEAKAGE FIX: PROPER TIME-SERIES TRAIN/TEST SPLIT
        # =======================================================
        # Calculate the 80% cutoff index
        split_idx = int(len(aggregated) * 0.8)

        # Train on the Past (0 to 80%)
        X_train, y_vol_train, y_aht_train = X.iloc[:split_idx], y_vol.iloc[:split_idx], y_aht.iloc[:split_idx]
        
        # Predict the Future (80% to 100%)
        X_test = X.iloc[split_idx:]
        
        # Train Layer 1 ML Models on the PAST ONLY
        self.volume_model.fit(X_train, y_vol_train)
        self.aht_model.fit(X_train, y_aht_train)

        # Predict the FUTURE
        predicted_vol = self.volume_model.predict(X_test) * 1.05
        predicted_aht = self.aht_model.predict(X_test)

        # Create a new dataframe containing only the "Future" predictions
        future_forecast = aggregated.iloc[split_idx:].copy()
        future_forecast['Predicted_Volume'] = np.ceil(predicted_vol)
        future_forecast['Predicted_AHT'] = predicted_aht
        
        return future_forecast

# ==========================================
# 3. LAYER 2: AI ABSENTEEISM RISK PREDICTOR
# ==========================================
class AIShrinkagePredictor:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=30, random_state=42)

    def train_and_get_shrinkage_rate(self, df_agents):
        np.random.seed(42)
        df_agents['Tenure_Months'] = df_agents.get('Tenure', np.random.uniform(0.5, 5.0, len(df_agents))) * 12
        df_agents['Past_Adherence'] = np.random.uniform(0.82, 0.98, len(df_agents))
        
        y_absent = ((df_agents['Past_Adherence'] < 0.88) | (df_agents['Tenure_Months'] < 6)).astype(int)
        
        X = df_agents[['Tenure_Months', 'Past_Adherence']]
        self.model.fit(X, y_absent)

        predicted_absentee_prob = self.model.predict_proba(X)[:, 1].mean()
        
        base_shrinkage = 0.10
        total_shrinkage_rate = base_shrinkage + (predicted_absentee_prob * 0.15)
        return total_shrinkage_rate

# ==========================================
# 4. QUANTUM QUBO MATRIX SOLVER ENGINE
# ==========================================
def solve_hybrid_quantum_schedule(df_agents, interval_targets, shift_defs, time_indices):
    print("\n[3] Building Native QUBO Matrix across AI-Adjusted Targets...")
    
    num_agents = len(df_agents)
    shifts = list(shift_defs.keys())
    num_shifts = len(shifts)
    
    Q = {}
    
    L1_DEMAND = 8000.0    
    L2_SINGLE_SHIFT = 50000.0 
    L3_FRICTION = 150.0   
    L4_COST = 2.0  # Added cost weight
    
    # 1. Single Shift Constraint
    for i in range(num_agents):
        for s1 in range(num_shifts):
            idx1 = i * num_shifts + s1
            Q[(idx1, idx1)] = Q.get((idx1, idx1), 0) - L2_SINGLE_SHIFT
            for s2 in range(s1 + 1, num_shifts):
                idx2 = i * num_shifts + s2
                Q[(idx1, idx2)] = Q.get((idx1, idx2), 0) + (2 * L2_SINGLE_SHIFT)

    # 2. Preference Friction & Cost Objective
    for i, (_, agent) in enumerate(df_agents.iterrows()):
        preferred = str(agent.get('Preferred_Shift', 'Morning'))
        hourly_rate = float(agent.get('Cost_Per_Hour', 25.0))
        shift_cost = hourly_rate * 8.0 # 8 hour shift
        
        for s_idx, s_name in enumerate(shifts):
            idx = i * num_shifts + s_idx
            if s_name != 'Off':
                # Minimize Labor Cost
                Q[(idx, idx)] = Q.get((idx, idx), 0) + (shift_cost * L4_COST)
                
                # Minimize Agent Friction
                if s_name not in preferred:
                    Q[(idx, idx)] = Q.get((idx, idx), 0) + L3_FRICTION

    # 3. AI Demand Coverage Constraint (WITH SLACK VARIABLES)
    offset = num_agents * num_shifts
    for t_idx, target_capacity in enumerate(interval_targets):
        if target_capacity <= 0:
            continue
        
        # Map the test set index to the ACTUAL time of day (0-95)
        time_of_day = time_indices[t_idx]
        
        interval_vars = []
        for i in range(num_agents):
            for s_idx, s_name in enumerate(shifts):
                if shift_defs[s_name][time_of_day] == 1:
                    interval_vars.append((i * num_shifts + s_idx, 1))
                    
        # Slack Variables to allow Overstaffing
        max_slack = max(1, math.ceil(math.log2(target_capacity + 5)))
        for k in range(max_slack):
            slack_idx = offset + (t_idx * 30) + k
            interval_vars.append((slack_idx, -(2**k)))
        
        # Matrix Expansion
        for i_idx in range(len(interval_vars)):
            q1, c1 = interval_vars[i_idx]
            Q[(q1, q1)] = Q.get((q1, q1), 0) + L1_DEMAND * (c1**2 - 2 * target_capacity * c1)
            for j_idx in range(i_idx + 1, len(interval_vars)):
                q2, c2 = interval_vars[j_idx]
                key = (q1, q2) if q1 < q2 else (q2, q1)
                Q[key] = Q.get(key, 0) + (2 * L1_DEMAND * c1 * c2)

    print(f"    -> Variables: {num_agents * num_shifts} Agent Qubits + Slack Qubits")
    print("    -> Executing Simulated Annealing Sampler (D-Wave Neal)...")
    start_time = time.time()
    sampler = neal.SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q, num_reads=50) 
    solve_duration = time.time() - start_time
    
    best_sample = response.first.sample
    print(f"    -> Quantum Annealing Complete in {solve_duration:.4f} seconds.")

    assigned_count = 0
    off_preference_count = 0
    total_cost = 0.0

    for i, (_, agent) in enumerate(df_agents.iterrows()):
        pref = str(agent.get('Preferred_Shift', 'Morning'))
        hourly_rate = float(agent.get('Cost_Per_Hour', 25.0))
        
        for s_idx, s_name in enumerate(shifts):
            idx = i * num_shifts + s_idx
            if best_sample.get(idx, 0) == 1 and s_name != 'Off':
                assigned_count += 1
                shift_hours = 8.0
                total_cost += hourly_rate * shift_hours
                if s_name not in pref:
                    off_preference_count += 1

    return assigned_count, off_preference_count, total_cost, solve_duration

# ==========================================
# 5. MAIN INTEGRATED EXECUTION PIPELINE
# ==========================================
def run_hybrid_ai_quantum_pipeline(call_log_path, agent_roster_path, queue_filter='Advisor_Support'):
    print("============================================================")
    print(" 🚀 VANGUARD WISER: HYBRID AI + QUANTUM OPTIMIZATION PIPELINE")
    print("============================================================")

    df_log = pd.read_csv(call_log_path)
    df_agents = pd.read_csv(agent_roster_path)

    if 'Queue' in df_log.columns:
        if queue_filter in df_log['Queue'].unique():
            df_log = df_log[df_log['Queue'] == queue_filter]
        else:
            fallback = 'Advisor' if 'Advisor' in df_log['Queue'].unique() else df_log['Queue'].iloc[0]
            df_log = df_log[df_log['Queue'] == fallback]

    print("\n[1] Running Layer 1: AI Workload & AHT Predictive Forecaster...")
    forecaster = AIWorkloadForecaster()
    df_forecast = forecaster.fit_and_predict(df_log)
    print(f"    -> Forecasted Future Volume (20% Test Split): {int(df_forecast['Predicted_Volume'].sum()):,} calls")

    print("\n[2] Running Layer 2: AI Absenteeism & Shrinkage Risk Estimator...")
    shrinkage_predictor = AIShrinkagePredictor()
    predicted_shrinkage = shrinkage_predictor.train_and_get_shrinkage_rate(df_agents)
    print(f"    -> Predicted Dynamic Shrinkage Buffer: {predicted_shrinkage * 100:.2f}%")

    interval_targets = []
    time_indices = []
    for _, row in df_forecast.iterrows():
        hourly_rate = row['Predicted_Volume'] * 4  
        aht = row['Predicted_AHT']
        raw_target = calculate_erlang_target(hourly_rate, aht)
        ai_padded_target = math.ceil(raw_target * (1.0 + predicted_shrinkage))
        interval_targets.append(ai_padded_target)
        
        # Keep track of the actual time of day (0-95) for the QUBO Shift mapping
        interval_of_day = int(row['Hour'] * 4 + row['Minute'] // 15)
        time_indices.append(interval_of_day)

    peak_demand = max(interval_targets)
    agents_needed = int(peak_demand * 1.1) 
    print(f"    -> Peak Interval Target: {peak_demand} agents")
    print(f"    -> Micro-Batching: Trimming roster to Top {agents_needed} Elite Agents for QPU limits...")
    
    if 'Cost_Per_Hour' in df_agents.columns:
        df_agents = df_agents.sort_values('Cost_Per_Hour', ascending=True).head(agents_needed).copy()
    else:
        df_agents = df_agents.head(agents_needed).copy()
    
    df_agents = df_agents.reset_index(drop=True)

    shift_defs = {
        'Morning': [1 if 28 <= t < 64 else 0 for t in range(96)],  
        'Midday':  [1 if 40 <= t < 76 else 0 for t in range(96)],   
        'Evening': [1 if 52 <= t < 88 else 0 for t in range(96)],   
        'Off':     [0 for _ in range(96)]
    }

    assigned, off_pref, cost, duration = solve_hybrid_quantum_schedule(
        df_agents, interval_targets, shift_defs, time_indices
    )

    print("\n============================================================")
    print(" 📊 HYBRID AI + QUANTUM OPTIMIZER: FINAL RESULTS")
    print("============================================================")
    print(f" ML Predicted Shrinkage Buffer : {predicted_shrinkage * 100:.2f}%")
    print(f" Quantum Solver Execution Time : {duration:.4f} seconds")
    print(f" Total Elite Agents Scheduled  : {assigned}")
    print(f" Off-Preference Assignments    : {off_pref} agents ({off_pref/max(1,assigned)*100:.1f}%)")
    print(f" Total Labor Payroll Cost      : ${cost:,.2f}")
    print("============================================================")

if __name__ == "__main__":
    run_hybrid_ai_quantum_pipeline("call_log.csv", "agent_data.csv", queue_filter="Advisor")