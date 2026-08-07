# Shiftwave: Architecture & File Guide

This document provides a detailed breakdown of the script files, experimental iterations, and expected console/interface outputs contained within the Shiftwave repository.

---
# Shiftwave: Architecture & File Guide

This document provides a detailed breakdown of the script files, experimental iterations, expected console/interface outputs, and instructions for running the primary Streamlit application (`sol.py`) contained within the Shiftwave repository.

---
Remaining Every Other code provided are the trials I have gone throughout o get to sol.py.

## 🌟 Primary Application: `sol.py`

`sol.py` serves as the core flagship program for Shiftwave. It bridges the mathematical QUBO energy landscape with an interactive web interface built using **Streamlit**, enabling real-time management of workforce parameters.

### How to Run the Streamlit App
To launch the interactive dashboard locally:

1. Ensure you have Streamlit, D-Wave Ocean SDK, and Pandas installed in your environment:
   ```bash
   pip install streamlit dwave-neal pandas numpy pulp scikit-learn

## 📂 Repository File Structure & Descriptions

### 1. `baseline.py`
*   **Description:** Establishes the baseline benchmark using traditional Integer Linear Programming (ILP) via the PuLP library and CBC Branch-and-Cut algorithm[cite: 1]. It filters elite agents and applies rigid equality constraints to meet Erlang C demand targets[cite: 1], including a 300-second timeout to prevent CPU lockups[cite: 1].
*   **Expected Output:**
    ```text
    [0] Loading Data for Classical Baseline Benchmark...
    [2] Loading and Filtering Agent Roster...
        -> Selected [N] Elite Agents for the Classical Solver.
    [1] Initializing Classical MIP Solver (PuLP) for [Target Queue]...
        -> Generating Binary Decision Variables...
        -> Formulating Objective Function (Cost + Preference Friction)...
        -> Applying Hard Constraints...

    [2] Solving Classical MIP... (This may take a moment)
        -> Initiating CBC Branch-and-Cut Algorithm...
        -> WARNING: Time limit set to 300 seconds (5 minutes) to prevent CPU lockup.

    ==================================================
     🖥️ CLASSICAL SOLVER RESULTS ([Solver Status])
    ==================================================
    Execution Time    : [Time] seconds
    Total Agents      : [Count]
    Total Labor Cost  : $[Cost]
    Off-Preference    : [Count] agents
    ==================================================
    ```

### 2. `multi_mod3.py`
*   **Description:** Introduces the "Dual-Engine" hybrid architecture, merging a macro Erlang C evaluator with a custom micro Discrete Event Simulator (DES)[cite: 2]. It applies High-Resolution BQM Quantum Assignment by clustering agents into micro-squads of 5 to balance execution speed and operational resolution[cite: 2].
*   **Expected Output:**
    ```text
    [1] Checking for datasets...
    [2] Processing Call Data & Generating Erlang C Targets for MONDAY...
    [MODULE A] Classical Phase: Calculating Temporal SLA Bounds...
    [MODULE B] Processing Trimmed Agent Pool (Micro-Clustering)...
    [MODULE B] Building Matrix and Expanding Algebra...
    [MODULE B] Running D-Wave Simulated Annealer (Balanced for Speed/Res)...
    [PIPELINE COMPLETE] Unpacking Agents...
    [DIAGNOSTIC] Running Discrete Event Simulation on Quantum Schedule...

    ===========================================================================
     📊 VANGUARD ENTERPRISE SCORECARD (DUAL-ENGINE EVALUATION)
    ===========================================================================
    --- MACRO METRICS (ERLANG C) ---
    Execution Time (sec)           | [Time]
    Total Agents Deployed          | [Count]
    Projected Payroll              | $[Cost]
    Service Level (SLA %)          | [SLA]%
    Avg Speed of Answer            | [ASA] s
    Queue Abandonment Rate         | [Abandon]%

    --- MICRO METRICS (EVENT SIMULATOR) ---
    Avg Idle Time Between Calls    | [Idle] mins
    Agents Who Took ZERO Calls     | [Count]
    Max Calls Taken by One Agent   | [Count]
    ===========================================================================

    🚨 EXTREME OVERWORK (Top 3 Busiest Agents):
    [DataFrame Table of Busiest Agents]

    💤 EXTREME IDLE TIME (Top 3 Least Busy Agents):
    [DataFrame Table of Idle Agents]
    ```

### 3. `sol.py`
*   **Description:** An evolution of the optimization dashboard featuring refined base shift preference logic, dynamic micro-shift costing, and an integrated Streamlit app[cite: 3]. It features Manager Controls via interactive sliders to dynamically adjust SLA priority, cost efficiency, and employee preference weights on the QUBO energy landscape[cite: 3].
*   **Expected Output (Streamlit UI Interface):**
    ```text
     Quantum Multi-Model Workforce Planner
    Dynamic Bipartite Matching via D-Wave Simulated Annealing

    --- [Sidebar Controls] ---
    Manager Controls (SLA Priority, Cost Efficiency Priority, Employee Preference Priority)
    [Button: 🚀 Run Quantum Optimization]

    --- [Main Screen Scorecards (Upon Execution)] ---
    Projected Payroll     | Service Level (SLA)      | Avg Speed of Answer     | Off-Preference Shifts
    $[Cost] (- vs Class)  | [SLA]% (- vs Class)      | [ASA] s                 | [Count]

    --- [Interval Staffing Planner Table] ---
    Time Interval | Erlang C Target | Quantum Scheduled | Variance
    [15-min rows color-coded by understaffed vs optimal status]

    --- [Trade-off Analysis Box] ---
    [Dynamic status info regarding current energy valley and cost-vs-service trade-offs]
    ```

### 4. `trial3.py`
*   **Description:** Executes a direct head-to-head evaluation between the Classical LP (PuLP) solver and the True Quantum Simulator[cite: 4]. It utilizes a Constrained Quadratic Model (CQM) flattened to a BQM, heavily compressing the matrix by grouping 12 agents per squad to scale enterprise data efficiently[cite: 4].
*   **Expected Output:**
    ```text
    [CLASSICAL] Executing PuLP Linear Programming...
    [QUANTUM] Clustering Agents into Macro-Qubits (Squads)...
        -> Compressed [N] agents into [N] manageable Qubits.
    [QUANTUM] Compiling True Constrained Quadratic Model (CQM)...
    [QUANTUM] Flattening CQM to Binary Quadratic Model (BQM)...
    [QUANTUM] Running D-Wave Simulated Annealer (neal)...
    [QUANTUM] Unpacking Macro-Qubits to Individual Agents...

    ===========================================================================
     📊 VANGUARD ENTERPRISE SCORECARD (CLASSICAL VS. QUANTUM-INSPIRED)
    ===========================================================================
    Operational Metric        | Classical ILP (PuLP)   | Quantum-Inspired
    ---------------------------------------------------------------------------
    Execution Time (sec)      | [Time]                 | [Time]
    Total Agents Deployed     | [Count]                | [Count]
    Projected Payroll         | $[Cost]                | $[Cost]
    Routing Mismatches        | [Count]                | [Count]
    Off-Preference Shifts     | [Count]                | [Count]
    Service Level (SLA %)     | [SLA]%                 | [SLA]%
    Avg Speed of Answer       | [ASA] s                | [ASA] s
    Queue Abandonment Rate    | [Abandon]%             | [Abandon]%
    ===========================================================================
    ```

### 5. `trial2.py`
*   **Description:** Represents the foundational 3D Hamiltonian formulation (Agent × Shift × Queue) designed for a Master's Capstone submission[cite: 5]. It applies squared slack variables for dynamic demand absorption and safely limits the simulated annealing matrix to a hard cap of 250 elite agents to prevent local CPU crashes[cite: 5].
*   **Expected Output:**
    ```text
    [1] Loading external datasets...
    [2] Processing Call Data & Generating Erlang C Targets...
    [3] Pre-processing Agent Pool...
        -> Dynamically scoped matrix to [N] agents.
    [QUANTUM LAYER] Compiling 3D Variables (Agent x Shift x Queue)...
    [QUANTUM LAYER] Compiling H_demand (Slack & Non-Linear Proficiency)...
    [QUANTUM LAYER] Initializing D-Wave Simulated Annealing...

    ======================================================================
     🚀 FINAL ENTERPRISE SCHEDULE (3D QUANTUM QUBO)
    ======================================================================
    ✅ [Agent_ID] (Tier [T] | Prof: [P]x) -> [Shift] on [Queue] [Mismatch Flag]
    ...
    ----------------------------------------------------------------------
    Total Annealing Time     : [Time] sec
    Total Agents Deployed    : [Count]
    Total Projected Payroll  : $[Cost]
    Sub-Optimal Routings     : [Count]
    ======================================================================
    ```

### 6. `last_benchmark.py`
*   **Description:** The full-scale, uncapped version of the 3D Hamiltonian model[cite: 6]. It bypasses CPU limits, passing the full mathematically required agent roster into a 300-second bounded classical solver and a high-sweep (1500 sweeps) quantum simulated annealer for a true enterprise stress test[cite: 6].
*   **Expected Output:**
    ```text
    [1] Loading external datasets...
    [2] Processing Call Data & Generating Erlang C Targets...
    [3] Pre-processing Agent Pool...
        -> Target Demand requires ~[N] agents.
        -> Dynamically scoped matrix to [N] agents (FULL POOL UNCAPPED).
    [CLASSICAL LAYER] Executing PuLP Linear Programming (Up to 5 min limit)...
    [QUANTUM LAYER] Compiling 3D Variables (Agent x Shift x Queue)...
    [QUANTUM LAYER] Compiling H_demand (Slack & Non-Linear Proficiency)...
    [QUANTUM LAYER] Initializing D-Wave Simulated Annealing...

    ===========================================================================
     📊 VANGUARD ENTERPRISE SCORECARD (FULL CAPACITY HEAD-TO-HEAD)
    ===========================================================================
    Operational Metric        | Classical ILP (PuLP)   | Quantum QUBO (neal)
    ---------------------------------------------------------------------------
    Execution Time (sec)      | [Time]                 | [Time]
    Total Agents Deployed     | [Count]                | [Count]
    Projected Payroll         | $[Cost]                | $[Cost]
    Routing Mismatches        | [Count]                | [Count]
    Off-Preference Shifts     | [Count]                | [Count]
    Service Level (SLA %)     | [SLA]%                 | [SLA]%
    Avg Speed of Answer       | [ASA] s                | [ASA] s
    Queue Abandonment Rate    | [Abandon]%             | [Abandon]%
    ===========================================================================
    ```

### 7. `HYBRID_QUBO.py`
*   **Description:** Explores decoupling static history by integrating Machine Learning, using `RandomForestRegressor` and `RandomForestClassifier` pipelines to forecast volume, AHT, and agent shrinkage[cite: 7]. It applies a strict time-series train/test split to prevent data leakage before generating the QUBO matrix[cite: 7].
*   **Expected Output:**
    ```text
    ============================================================
     🚀 VANGUARD WISER: HYBRID AI + QUANTUM OPTIMIZATION PIPELINE
    ============================================================
    [1] Running Layer 1: AI Workload & AHT Predictive Forecaster...
        -> Forecasted Future Volume (20% Test Split): [N] calls
    [2] Running Layer 2: AI Absenteeism & Shrinkage Risk Estimator...
        -> Predicted Dynamic Shrinkage Buffer: [Rate]%
    [3] Building Native QUBO Matrix across AI-Adjusted Targets...
        -> Variables: [N] Agent Qubits + Slack Qubits
        -> Executing Simulated Annealing Sampler (D-Wave Neal)...
        -> Quantum Annealing Complete in [Time] seconds.

    ============================================================
     📊 HYBRID AI + QUANTUM OPTIMIZER: FINAL RESULTS
    ============================================================
     ML Predicted Shrinkage Buffer : [Rate]%
     Quantum Solver Execution Time : [Time] seconds
     Total Elite Agents Scheduled  : [Count]
     Off-Preference Assignments    : [Count] agents ([Pct]%)
     Total Labor Payroll Cost      : $[Cost]
    ============================================================
    ```

### 8. `qubo1.py`
*   **Description:** A specialized QUBO matrix builder explicitly targeting high-complexity, lower-volume skill queues like 'Advisor'[cite: 8]. It enforces strict one-hot validity constraints (sum ≤ 1) and filters specifically for elite, full-time, highly tenured agents to feed the simulated annealer[cite: 8].
*   **Expected Output:**
    ```text
    [1] Loading Event Log for [Date] | Target Queue: [Queue]...
        -> AI Deflection applied. Remaining Human Calls: [N]
        -> Erlang C Peak Headcount Required: [N] Agents.
    [2] Loading and Filtering Agent Roster...
        -> Selected [N] Elite/Tenured Agents with [Queue] skills.
    [3] Building QUBO Matrix for [N] Elite agents...
        -> QUBO Matrix successfully built with [N] active couplings.
    [4] Initializing D-Wave Simulated Annealer...

    ============================================================
     🚀 OPTIMIZED SKILL-BASED SCHEDULE: [QUEUE]
    ============================================================
    ✅ [Agent_ID] (Tenure: [Value]) -> [Shift] | Cost: $[Cost] [Preference Flag]
    ...
    ------------------------------------------------------------
    Total Elite Agents Assigned: [Count]
    Rule Violations (Cloned Agents): 0 (Must be 0)
    Total Specialized Labor Cost: $[Cost]
    ============================================================
    ```
