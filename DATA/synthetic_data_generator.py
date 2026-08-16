import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta
import json

def get_interval_weight(day_of_week, hour):
    """Returns the raw statistical weight for a specific hour based on the day of the week."""
    if day_of_week == 0:  # Monday: Bi-modal M-Curve
        if 7 <= hour <= 19:
            morning = 1.5 * np.exp(-0.5 * ((hour - 9) / 1.5)**2)
            afternoon = 0.8 * np.exp(-0.5 * ((hour - 14) / 2.0)**2)
            return morning + afternoon + 0.2
        return 0.05

    elif day_of_week in [1, 2, 3]:  # Tue/Wed/Thu: Standard broad peak
        if 7 <= hour <= 19:
            return np.exp(-0.5 * ((hour - 11.5) / 3.5)**2) + 0.15
        return 0.05

    elif day_of_week == 4:  # Friday: Left-skewed (busy morning, dead afternoon)
        if 7 <= hour <= 18:
            return np.exp(-0.5 * ((hour - 10) / 2.5)**2) + 0.1
        return 0.05

    else:  # Weekend: Narrow window, flatter curve
        if 9 <= hour <= 15:
            return np.exp(-0.5 * ((hour - 12) / 2.0)**2) + 0.1
        return 0.01

def generate_data(start_date_str='2026-08-03'):
    start_date = datetime.strptime(start_date_str, '%Y-%m-%d')
    days = 7
    
    np.random.seed(42)
    
    queue_configs = {
        'General_Support': {'base_vol': 60000, 'ai_rate': 0.50, 'aht_mean': 840, 'aht_sig': 0.35},
        'Advisor':         {'base_vol': 25000, 'ai_rate': 0.30, 'aht_mean': 1020, 'aht_sig': 0.40},
        'Investor_Svc':    {'base_vol': 12000, 'ai_rate': 0.15, 'aht_mean': 1140, 'aht_sig': 0.40},
        'Trust_Intermed':  {'base_vol': 3000,  'ai_rate': 0.05, 'aht_mean': 1500, 'aht_sig': 0.45}
    }
    
    # Daily volume multipliers to average ~75k calls/day
    daily_multipliers = {0: 1.25, 1: 1.0, 2: 0.95, 3: 0.90, 4: 0.85, 5: 0.25, 6: 0.10}
    
    event_log = []
    call_id_counter = 1
    
    for day in range(days):
        current_day_date = start_date + timedelta(days=day)
        dow = current_day_date.weekday()
        
        # 1. Normalize the diurnal curve for this specific day
        raw_weights = []
        for interval in range(96):
            hour = (interval * 15) / 60.0
            raw_weights.append(get_interval_weight(dow, hour))
            
        sum_weights = sum(raw_weights)
        normalized_shares = [w / sum_weights for w in raw_weights]
        
        # 2. Generate calls for each 15-minute block
        for interval in range(96):
            interval_start = current_day_date + timedelta(minutes=15 * interval)
            interval_end = interval_start + timedelta(minutes=15)
            
            interval_share = normalized_shares[interval]
            
            for q_name, config in queue_configs.items():
                # Target volume for this queue, on this day, in this 15-min block
                target_daily_vol = config['base_vol'] * daily_multipliers[dow]
                expected_calls = target_daily_vol * interval_share
                
                rate_per_second = expected_calls / 900.0 
                
                if rate_per_second <= 0:
                    continue
                    
                current_time = interval_start
                
                # Exponential inter-arrival simulation
                while True:
                    gap_seconds = np.random.exponential(scale=1.0 / rate_per_second)
                    current_time += timedelta(seconds=gap_seconds)
                    
                    if current_time >= interval_end:
                        break 
                    
                    # AI Deflection
                    is_human = np.random.random() > config['ai_rate']
                    handled_by = 'Human' if is_human else 'AI_Bot'
                    
                    # Lognormal Handle Time
                    if is_human:
                        mu = np.log(config['aht_mean']) - (config['aht_sig']**2 / 2)
                        aht = round(np.random.lognormal(mu, config['aht_sig']), 1)
                    else:
                        aht = 0.0
                        
                    event_log.append({
                        'Call_ID': f'CALL_{call_id_counter:07d}',
                        'Arrival_Timestamp': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                        'Day_of_Week': current_time.strftime('%A'),
                        'Queue': q_name,
                        'Handled_By': handled_by,
                        'Handle_Time_Seconds': aht
                    })
                    call_id_counter += 1

    df_events = pd.DataFrame(event_log)
    df_events = df_events.sort_values(by='Arrival_Timestamp').reset_index(drop=True)
    df_events.to_csv('call_log.csv', index=False)
    
    # 2. AGENT DETAILS GENERATION (Added Part-Time Contract Types, Shift Preference etc..)
    
    agent_tiers = [
        {'title': 'General',      'w_min': 18.5, 'w_max': 22.0, 'share': 0.50, 'skills': [1,0,0,0]},
        {'title': 'Advisor',      'w_min': 22.0, 'w_max': 27.0, 'share': 0.25, 'skills': [1,1,0,0]},
        {'title': 'Investor_Svc', 'w_min': 25.0, 'w_max': 28.0, 'share': 0.15, 'skills': [0,1,1,0]},
        {'title': 'Intermed',     'w_min': 27.0, 'w_max': 30.0, 'share': 0.10, 'skills': [0,0,1,1]}
    ]
    
    contract_types = [
        {'type': 'Full-Time', 'prob': 0.70, 'elapsed': 9.0, 'max_wk': 40, 'cap': 0.833},
        {'type': 'PT-Heavy',  'prob': 0.20, 'elapsed': 6.0, 'max_wk': 30, 'cap': 0.850},
        {'type': 'PT-Peak',   'prob': 0.10, 'elapsed': 4.0, 'max_wk': 20, 'cap': 0.900}
    ]
    
    shift_prefs = ['Morning (06:00-14:00)', 'Midday (10:00-18:00)', 'Evening (14:00-22:00)']
    
    agent_data = []
    agent_id = 1
    total_agents = 10000
    
    for tier in agent_tiers:
        count = int(total_agents * tier['share'])
        for _ in range(count):
            wage = round(np.random.uniform(tier['w_min'], tier['w_max']), 2)
            base_tenure = 1.0 if tier['title'] == 'General' else 0.90
            tenure_mult = np.clip(round(np.random.normal(base_tenure, 0.1), 2), 0.80, 1.25)
            
            # Select contract type
            contract = random.choices(contract_types, weights=[c['prob'] for c in contract_types], k=1)[0]
            
            agent_data.append({
                'Agent_ID': f'AGT_{agent_id:05d}',
                'Tier': tier['title'],
                'Contract_Type': contract['type'],
                'Hourly_Wage': wage,
                'Tenure_AHT_Multiplier': tenure_mult,
                'Skill_GenSupport': tier['skills'][0],
                'Skill_Advisor': tier['skills'][1],
                'Skill_Investor': tier['skills'][2],
                'Skill_Intermed': tier['skills'][3],
                'Preferred_Shift': random.choice(shift_prefs),
                'Shift_Elapsed_Hours': contract['elapsed'],
                'Max_Weekly_Hours': contract['max_wk'],
                'Micro_Break_Capacity_Multiplier': contract['cap']
            })
            agent_id += 1
            
    df_agents = pd.DataFrame(agent_data)
    df_agents = df_agents.sample(frac=1, random_state=42).reset_index(drop=True)
    df_agents.to_csv('agent_DATA.csv', index=False)
    
    print(f"Generated Event Log with {len(df_events):,} individual calls.")
    print(f"Generated Roster with {len(df_agents):,} mixed-contract agents.")
    
    #3. CONSTRAINTS & PENALTIES
    
    business_rules = {
        "service_level_agreements": {
            "target_percentage": 0.80,
            "threshold_seconds": 20
        },
        "optimization_weights": {
            "penalty_sla_failure": 1000,
            "penalty_dol_violation": 5000,
            "penalty_preference_violation": 50,  # Schedule friction penalty
            "reward_cost_efficiency": 10
        }
    }
    
    with open('business_rules.json', 'w') as f:
        json.dump(business_rules, f, indent=4)
        

if __name__ == "__main__":
    generate_data()
