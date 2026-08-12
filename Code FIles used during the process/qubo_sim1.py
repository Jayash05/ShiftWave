import pandas as pd
import numpy as np
import math
from itertools import combinations
from datetime import datetime
from neal import SimulatedAnnealingSampler


# 1. ERLANG C CALCULATION
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

# 2. DATA INGESTION
def load_specialized_queue(events_file, agents_file, target_date, target_queue):
    print(f"\n[1] Loading Event Log for {target_date} | Target Queue: {target_queue}...")
    
    # Data Prep
    df_events = pd.read_csv(events_file)
    df_events['Arrival_Timestamp'] = pd.to_datetime(df_events['Arrival_Timestamp'])
    
    # FILTER 1: Target Date, Target Queue, and HUMAN ONLY
    mask = (df_events['Arrival_Timestamp'].dt.date.astype(str) == target_date) & \
           (df_events['Queue'] == target_queue) & \
           (df_events['Handled_By'] == 'Human')
           
    human_calls = df_events[mask].copy()
    
    if len(human_calls) == 0:
        print(f"    [!] WARNING: 0 Human calls found for {target_queue} on {target_date}.")
        fallback_queue = 'General_Support'
        print(f"    [!] Falling back to '{fallback_queue}' to demonstrate the optimizer.")
        
        mask = (df_events['Arrival_Timestamp'].dt.date.astype(str) == target_date) & \
               (df_events['Queue'] == fallback_queue) & \
               (df_events['Handled_By'] == 'Human')
        human_calls = df_events[mask].copy()
        target_queue = fallback_queue
        
        if len(human_calls) == 0:
             print("    [!] ERROR: Still 0 calls found. Please check your CSV dates and queue names.")
             return None, None, target_queue

    print(f" AI Deflection applied. Remaining Human Calls: {len(human_calls)}")
    # Aggregate into 15-min intervals (No artificial 5% scaling needed for specialized queues)
    human_calls.set_index('Arrival_Timestamp', inplace=True)
    interval_stats = human_calls.resample('15Min').agg(
        Volume=('Call_ID', 'count'),
        AHT=('Handle_Time_Seconds', 'mean')
    ).fillna(0)
    
    interval_stats['Target_Headcount'] = interval_stats.apply(
        lambda row: calculate_required_agents(row['Volume'], row['AHT']), axis=1
    )
    
    max_needed = interval_stats['Target_Headcount'].max()
    print(f"    -> Erlang C Peak Headcount Required: {max_needed} Agents.")
    
    print("\n[2] Loading and Filtering Agent Roster...")
    df_agents = pd.read_csv(agents_file)
    
    # We can grab slightly more agents than the peak required to give the solver flexibility
    skill_col = f"Skill_{target_queue.replace('_Support', '').replace('_Svc', '')}"
    
    # Find agents with the skill
    skilled_agents = df_agents[df_agents[skill_col] == 1].copy()
    
    # Sort by Contract Type (Full-Time first) and then by Tenure (Highest first)
    skilled_agents['Is_FT'] = (skilled_agents['Contract_Type'] == 'Full-Time').astype(int)
    elite_agents = skilled_agents.sort_values(
        by=['Is_FT', 'Tenure_AHT_Multiplier'], 
        ascending=[False, False]
    ).head(int(max_needed * 1.5)) # Buffer pool
    
    print(f"Selected {len(elite_agents)} Elite/Tenured Agents with {target_queue} skills.")
    
    return interval_stats, elite_agents


# 3. QUBO MATRIX GENERATION
def build_qubo(interval_stats, df_agents, target_date_str):
    print(f"\n[3] Building QUBO Matrix for {len(df_agents)} Elite agents...")
    target_day_name = pd.to_datetime(target_date_str).strftime('%A')
    
    shift_starts = {'Morning': 6, 'Midday': 10, 'Evening': 14}
    
    # Balanced Hyperparameters
    L1_DEMAND = 50          
    L2_VALID  = 50000       
    L3_FRICTION = 5         
    
    Q = {} 
    def add_weight(var1, var2, weight):
        k = tuple(sorted([var1, var2]))
        Q[k] = Q.get(k, 0) + weight

    for _, agent in df_agents.iterrows():
        a_id = agent['Agent_ID']
        wage = agent['Hourly_Wage']
        duration = agent['Shift_Elapsed_Hours']
        pref_shift = agent['Preferred_Shift']
        unavail_day = agent.get('Unavailable_Day', 'None')
        
        agent_vars = [f"x_{a_id}_{s_name}" for s_name in shift_starts.keys()]
        
        # 1. HARD VALIDITY (One-Hot: sum(x) <= 1)
        for v in agent_vars:
            add_weight(v, v, -L2_VALID) 
        for v1, v2 in combinations(agent_vars, 2):
            add_weight(v1, v2, 2 * L2_VALID)
            
        # 2. Linear Costs & Friction
        for shift_name in shift_starts.keys():
            var_name = f"x_{a_id}_{shift_name}"
            cost = wage * duration
            friction = 0
            if target_day_name == unavail_day: friction += 2 
            if shift_name not in pref_shift: friction += 1 
                
            add_weight(var_name, var_name, cost + (L3_FRICTION * friction))

    # 3. Demand Penalty with Slack Variables
    for t in interval_stats[interval_stats['Target_Headcount'] > 0].index:
        target_N = interval_stats.loc[t, 'Target_Headcount']
        hour_val = t.hour + t.minute / 60.0
        
        active_vars = []
        for _, agent in df_agents.iterrows():
            a_id = agent['Agent_ID']
            duration = agent['Shift_Elapsed_Hours']
            prof = agent['Tenure_AHT_Multiplier'] * agent['Micro_Break_Capacity_Multiplier']
            
            for shift_name, start_hr in shift_starts.items():
                if start_hr <= hour_val < (start_hr + duration):
                    active_vars.append((f"x_{a_id}_{shift_name}", prof))
                    
        max_slack = max(1, math.ceil(math.log2(target_N + 5))) if target_N > 0 else 1
        slack_vars = [(f"y_{t.strftime('%H%M')}_{k}", 2**k) for k in range(max_slack)]
        
        interval_vars = active_vars + [(name, -coeff) for name, coeff in slack_vars]
        
        for var_name, coeff in interval_vars:
            add_weight(var_name, var_name, L1_DEMAND * (coeff**2 - 2 * target_N * coeff))
        for (var1, coeff1), (var2, coeff2) in combinations(interval_vars, 2):
            add_weight(var1, var2, L1_DEMAND * 2 * coeff1 * coeff2)

    print(f"    -> QUBO Matrix successfully built with {len(Q)} active couplings.")
    return Q

# 4. SOLVER INTEGRATION
def solve_with_dwave(Q, df_agents, target_queue):
    if SimulatedAnnealingSampler is None:
        print("\n[!] ERROR: 'dwave-neal' is not installed.")
        return None
        
    print("\n[4] Initializing D-Wave Simulated Annealer...")
    sampler = SimulatedAnnealingSampler()
    response = sampler.sample_qubo(Q, num_reads=100)
    best_sample = response.first.sample
    
    print(f" OPTIMIZED SKILL-BASED SCHEDULE: {target_queue.upper()}")
    
    scheduled_shifts = 0
    total_cost = 0.0
    agent_counts = {}
    shift_starts = {'Morning': 6, 'Midday': 10, 'Evening': 14}
    
    # Store the actual scheduled agents per interval
    scheduled_capacity = {i: 0 for i in range(96)} # 96 15-min intervals in a day

    for var_name, binary_value in best_sample.items():
        if binary_value == 1 and var_name.startswith('x_'):
            parts = var_name.split('_')
            agent_id = f"{parts[1]}_{parts[2]}"
            shift_name = parts[3]
            
            agent_counts[agent_id] = agent_counts.get(agent_id, 0) + 1
            agent_row = df_agents[df_agents['Agent_ID'] == agent_id].iloc[0]
            cost = agent_row['Hourly_Wage'] * agent_row['Shift_Elapsed_Hours']
            
            total_cost += cost
            scheduled_shifts += 1
            
            pref_flag = " [ Off-Preference]" if shift_name not in agent_row['Preferred_Shift'] else ""
            print(f" {agent_id} (Tenure: {agent_row['Tenure_AHT_Multiplier']}) -> {shift_name} | Cost: ${cost:.2f}{pref_flag}")
            
            # Update the scheduled_capacity dictionary
            start_hr = shift_starts[shift_name]
            duration = agent_row['Shift_Elapsed_Hours']
            start_interval = int(start_hr * 4)
            end_interval = int((start_hr + duration) * 4)
            
            for i in range(start_interval, end_interval):
                scheduled_capacity[i] += 1
    
    cloned_agents = sum(1 for counts in agent_counts.values() if counts > 1)
    print(f"Total Elite Agents Assigned: {scheduled_shifts}")
    print(f"Rule Violations (Cloned Agents): {cloned_agents} (Must be 0)")
    print(f"Total Specialized Labor Cost: ${total_cost:.2f}")
    
    return scheduled_capacity

# 5. DISCRETE EVENT SIMULATION
class VirtualCallCenter:
    def __init__(self, env, initial_agents):
        self.env = env
        self.agents = simpy.Resource(env, capacity=initial_agents)
        self.metrics = []

    def handle_call(self, call_id, handle_time, patience):
        arrival_time = self.env.now
        
        with self.agents.request() as request:
            results = yield request | self.env.timeout(patience)
            wait_time = self.env.now - arrival_time
            
            if request in results:
                self.metrics.append({
                    'call_id': call_id,
                    'wait_time': wait_time,
                    'abandoned': False,
                    'handled': True
                })
                yield self.env.timeout(handle_time)
            else:
                self.metrics.append({
                    'call_id': call_id,
                    'wait_time': wait_time,
                    'abandoned': True,
                    'handled': False
                })

def shift_manager(env, call_center, scheduled_capacity):
    """Dynamically changes the number of agents logged in every 15 minutes."""
    while True:
        current_interval = int(env.now // 900) 
        if current_interval in scheduled_capacity:
            # Capacity must be at least 1 to avoid SimPy errors
            new_capacity = max(1, scheduled_capacity[current_interval])
            
            if new_capacity != call_center.agents.capacity:
                call_center.agents._capacity = new_capacity
                
        yield env.timeout(900) 

def call_generator(env, call_center, df_calls):
    """Feeds historical calls into the simulation based on timestamps."""
    last_arrival = 0
    
    for _, row in df_calls.iterrows():
        seconds_from_midnight = (row.name.hour * 3600) + \
                                (row.name.minute * 60) + \
                                row.name.second
                                
        inter_arrival_time = max(0, seconds_from_midnight - last_arrival)
        last_arrival = seconds_from_midnight
        
        yield env.timeout(inter_arrival_time)
        
        # Caller patience (Avg 5 minutes before hanging up)
        patience = np.random.exponential(300) 
        
        env.process(call_center.handle_call(
            call_id=row['Call_ID'],
            handle_time=row['Handle_Time_Seconds'],
            patience=patience
        ))

def run_simulation(human_calls, scheduled_capacity):
    print("\n[5] Running SimPy Discrete Event Simulation...")
    print("Virtual agents are logging in and taking calls based on the optimized schedule...")
    
    env = simpy.Environment()
    call_center = VirtualCallCenter(env, initial_agents=max(1, scheduled_capacity.get(0, 1)))
    
    env.process(shift_manager(env, call_center, scheduled_capacity))
    env.process(call_generator(env, call_center, human_calls))
    
    # Force the simulation to stop at midnight (24 hours * 60 mins * 60 secs = 86400)
    env.run(until=86400) 
    
    print(" VIRTUAL CALL CENTER: END OF DAY REPORT")

    
    results = pd.DataFrame(call_center.metrics)
    total_calls = len(results)
    
    if total_calls == 0:
        print("No calls were processed in the simulation.")
        return
        
    handled_calls = len(results[results['handled'] == True])
    abandoned_calls = len(results[results['abandoned'] == True])
    
    sla_met = len(results[(results['handled'] == True) & (results['wait_time'] <= 20)])
    sla_percentage = (sla_met / total_calls) * 100
    
    asa = results[results['handled'] == True]['wait_time'].mean()
    
    print(f"Total Human Calls Processed : {total_calls:,}")
    print(f"Total Calls Abandoned       : {abandoned_calls:,} ({(abandoned_calls/total_calls)*100:.1f}%)")
    print(f"Average Speed of Answer     : {asa:.1f} seconds")
    print("-" * 50)
    
    if sla_percentage >= 80:
        print(f" SERVICE LEVEL AGREEMENT (SLA) MET: {sla_percentage:.1f}%")
        print("   (Target: 80% answered in 20 seconds)")
    else:
        print(f" SLA FAILED: {sla_percentage:.1f}%")
        print("   (Target: 80% answered in 20 seconds)")
    print("="*50)


if __name__ == "__main__":
    try:
        import simpy
        target_monday = '2026-08-03' 
        target_queue = 'Advisor' 
        
        stats, agents, final_queue = load_specialized_queue(
            'call_log.csv', 
            'agent_data.csv', 
            target_monday, 
            target_queue
        )
        
        if stats is not None:
             Q_matrix = build_qubo(stats, agents, target_monday)
             scheduled_capacity = solve_with_dwave(Q_matrix, agents, final_queue)
             
             # Re-load events specifically for the simulation generator to ensure it works properly
             df_events = pd.read_csv('call_log.csv')
             df_events['Arrival_Timestamp'] = pd.to_datetime(df_events['Arrival_Timestamp'])
             mask = (df_events['Arrival_Timestamp'].dt.date.astype(str) == target_monday) & \
                    (df_events['Queue'] == final_queue) & \
                    (df_events['Handled_By'] == 'Human')
             human_calls = df_events[mask].copy()
             human_calls.set_index('Arrival_Timestamp', inplace=True)
             human_calls.sort_index(inplace=True)
             
             run_simulation(human_calls, scheduled_capacity)
             
    except FileNotFoundError:
        print("ERROR: Could not find the CSV files.")
    except ImportError as e:
        print(f"Import Error: {e}. Please ensure simpy is installed: pip install simpy")
