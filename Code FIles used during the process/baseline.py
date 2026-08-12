import pandas as pd
import pulp
import time
import math

def solve_classical_mip(interval_stats, df_agents, target_queue):
    print(f"\n[1] Initializing Classical MIP Solver (PuLP) for {target_queue}...")
    
    # Create the linear programming problem (Minimization)
    prob = pulp.LpProblem("Vanguard_WISER_Scheduling", pulp.LpMinimize)
    
    shift_starts = {'Morning': 6, 'Midday': 10, 'Evening': 14}
    agent_vars = {}
    
    # 1. CREATE DECISION VARIABLES
    print("    -> Generating Binary Decision Variables...")
    for _, agent in df_agents.iterrows():
        a_id = agent['Agent_ID']
        for shift_name in shift_starts.keys():
            # x[i, j] = 1 if agent i is assigned to shift j, else 0
            var_name = f"x_{a_id}_{shift_name}"
            agent_vars[(a_id, shift_name)] = pulp.LpVariable(var_name, cat='Binary')
            
    # 2. OBJECTIVE FUNCTION: Minimize Cost + Friction
    print("    -> Formulating Objective Function (Cost + Preference Friction)...")
    objective_terms = []
    
    # Hyperparameter to scale friction against real dollars
    FRICTION_PENALTY_DOLLARS = 25 
    
    for _, agent in df_agents.iterrows():
        a_id = agent['Agent_ID']
        wage = agent['Hourly_Wage']
        duration = agent['Shift_Elapsed_Hours']
        pref_shift = agent['Preferred_Shift']
        
        for shift_name in shift_starts.keys():
            var = agent_vars[(a_id, shift_name)]
            cost = wage * duration
            
            friction = 0
            if shift_name not in pref_shift:
                friction += 1 
                
            total_weight = cost + (friction * FRICTION_PENALTY_DOLLARS)
            objective_terms.append(total_weight * var)
            
    prob += pulp.lpSum(objective_terms)
    
    # 3. CONSTRAINTS
    print("    -> Applying Hard Constraints...")
    
    # Constraint A: Validity (An agent can only work a maximum of 1 shift per day)
    for a_id in df_agents['Agent_ID'].unique():
        prob += pulp.lpSum([agent_vars[(a_id, s)] for s in shift_starts.keys()]) <= 1, f"MaxOneShift_{a_id}"

    # Constraint B: Demand (Must meet Erlang C requirement for every interval)
    for t in interval_stats[interval_stats['Target_Headcount'] > 0].index:
        target_N = interval_stats.loc[t, 'Target_Headcount']
        hour_val = t.hour + t.minute / 60.0
        
        active_agents_in_interval = []
        for _, agent in df_agents.iterrows():
            a_id = agent['Agent_ID']
            duration = agent['Shift_Elapsed_Hours']
            prof = agent['Tenure_AHT_Multiplier']
            
            for shift_name, start_hr in shift_starts.items():
                if start_hr <= hour_val < (start_hr + duration):
                    active_agents_in_interval.append(prof * agent_vars[(a_id, shift_name)])
                    
        # The sum of proficiencies of working agents must be >= Target Headcount
        if active_agents_in_interval:
            prob += pulp.lpSum(active_agents_in_interval) >= target_N, f"Demand_{t.strftime('%H%M')}"
            
    # 4. SOLVE
    print("\n[2] Solving Classical MIP... (This may take a moment)")
    print("    -> Initiating CBC Branch-and-Cut Algorithm...")
    print("    -> WARNING: Time limit set to 300 seconds (5 minutes) to prevent CPU lockup.")
    start_time = time.time()
    
    # Use CBC (Coin-or branch and cut), PuLP's default solver. 
    # Added a 300-second timeout so it doesn't run forever.
    prob.solve(pulp.PULP_CBC_CMD(msg=True, timeLimit=300))
    
    execution_time = time.time() - start_time
    print(f" CLASSICAL SOLVER RESULTS ({pulp.LpStatus[prob.status]})")
    
    total_cost = 0
    assigned_agents = 0
    off_preference = 0
    
    for (a_id, shift_name), var in agent_vars.items():
        if pulp.value(var) == 1.0:
            agent_row = df_agents[df_agents['Agent_ID'] == a_id].iloc[0]
            cost = agent_row['Hourly_Wage'] * agent_row['Shift_Elapsed_Hours']
            total_cost += cost
            assigned_agents += 1
            
            if shift_name not in agent_row['Preferred_Shift']:
                off_preference += 1
                
    print(f"Execution Time    : {execution_time:.4f} seconds")
    print(f"Total Agents      : {assigned_agents}")
    print(f"Total Labor Cost  : ${total_cost:.2f}")
    print(f"Off-Preference    : {off_preference} agents")
    
    return prob

def erlang_c(A, N):
    if N <= A: return 1.0 
    inv_b = 1.0
    for i in range(1, int(N) + 1):
        inv_b = 1.0 + (inv_b * i) / A
    erlang_b = 1.0 / inv_b
    p_w = (N * erlang_b) / (N - A * (1 - erlang_b))
    return min(max(p_w, 0.0), 1.0)

def calculate_required_agents(volume, aht, target_sl=0.80, target_time=20, interval_seconds=900):
    if volume == 0: return 0
    A = (volume * aht) / interval_seconds
    N = max(1, math.ceil(A))
    while True:
        p_w = erlang_c(A, N)
        sl = 1 - (p_w * math.exp(-(N - A) * (target_time / aht)))
        if sl >= target_sl: return N
        N += 1

if __name__ == "__main__":
    print("\n[0] Loading Data for Classical Baseline Benchmark...")
    try:
        df_events = pd.read_csv('call_log.csv')
        df_events['Arrival_Timestamp'] = pd.to_datetime(df_events['Arrival_Timestamp'])
        
        target_queue = 'Advisor' 
        target_date = '2026-08-03'
        
        mask = (df_events['Arrival_Timestamp'].dt.date.astype(str) == target_date) & \
               (df_events['Queue'] == target_queue) & \
               (df_events['Handled_By'] == 'Human')
        human_calls = df_events[mask].copy()
        human_calls.set_index('Arrival_Timestamp', inplace=True)
        
        interval_stats = human_calls.resample('15Min').agg(
            Volume=('Call_ID', 'count'),
            AHT=('Handle_Time_Seconds', 'mean')
        ).fillna(0)
        
        interval_stats['Target_Headcount'] = interval_stats.apply(
            lambda row: calculate_required_agents(row['Volume'], row['AHT']), axis=1
        )
        
        print("\n[2] Loading and Filtering Agent Roster...")
        df_agents = pd.read_csv('agent_data.csv') 
        """
        -------------------------------------------------------------
        APPLES-TO-APPLES BENCHMARK:
        Feed the exact same pool of Elite Agents to the Classical
        solver was fed to the Quantum Solver.
        -------------------------------------------------------------
        """
        skill_col = f"Skill_{target_queue.replace('_Support', '').replace('_Svc', '')}"
        if skill_col not in df_agents.columns:
            skill_cols = [c for c in df_agents.columns if c.startswith('Skill_')]
            skill_col = skill_cols[0] if skill_cols else 'Skill_GenSupport'

        skilled_agents = df_agents[df_agents[skill_col] == 1].copy()
        skilled_agents['Is_FT'] = (skilled_agents['Contract_Type'] == 'Full-Time').astype(int)
        
        max_needed = interval_stats['Target_Headcount'].max()
        
        elite_agents = skilled_agents.sort_values(
            by=['Is_FT', 'Tenure_AHT_Multiplier'], 
            ascending=[False, False]
        ).head(int(max_needed * 1.5))
        
        print(f"Selected {len(elite_agents)} Elite Agents for the Classical Solver.")
        
        solve_classical_mip(interval_stats, elite_agents, target_queue)
        
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
