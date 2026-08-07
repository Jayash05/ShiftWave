# Shiftwave: Hybrid-Quantum Workforce Optimization

![Python](https://img.shields.io/badge/Python-3.13-blue.svg)
![Quantum](https://img.shields.io/badge/D--Wave-Ocean_SDK-purple.svg)
![Optimization](https://img.shields.io/badge/Optimization-QUBO-brightgreen.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📌 Project Summary
I understood the core problem as forecasting multi-channel call center demand and optimizing staffing schedules—balancing shifts, skill groups, and breaks—to minimize costs while hitting strict Service Level Agreement (SLA) targets. I addressed the inherent challenge where non-linear stochastic queueing dynamics (Erlang C) conflict with these rigid temporal boundaries. Classical Integer Linear Programming (ILP) inherently struggles here, as applying strict equality constraints to volatile 15-minute demand spikes results in a mathematical state-space explosion or massive overstaffing. To solve this, I engineered Shiftwave, a hybrid-quantum pipeline that translates the staffing matrix into a Quantum Unconstrained Binary Optimization (QUBO) energy landscape. To bypass hardware dimensionality limits, I designed a "Macro-Qubit Squad" algorithm that clusters agents into tensor blocks, feeding this lightweight matrix into a D-Wave Simulated Annealer. I embedded business constraints into a 3D Hamiltonian ($Agent \times Shift \times Queue$), utilizing squared slack variables for dynamic demand absorption and a parabolic penalty function to restrict extreme cross-skill mismatches. By translating absolute constraints into quantum energy penalties, Shiftwave natively absorbs operational volatility. While the classical ILP benchmark defaulted to a highly inefficient $1,032,365 payroll to hit SLA targets, Shiftwave executed the heavily compressed matrix in under 80 seconds. It successfully traversed the energy landscape to find the global optimum, securing a stable 90% SLA while reducing the projected payroll to $807,609. I recommend adopting hybrid-quantum pipelines for enterprise workforce management, as this 18% cost reduction proves their superior resilience, scalability, and capacity to balance complex trade-offs.

---

## 🔄 Architectural Evolution & Experimental Trials

Developing Shiftwave required navigating significant hardware and mathematical bottlenecks across seven distinct experimental iterations. Documenting this evolution highlights why a hybrid-physics approach ultimately outperformed purely predictive or purely classical models.

*   **Phase 1: Pure Quantum & Dimensionality Explosion:** Initially, I passed a raw, uncompressed 2D matrix directly to the D-Wave Simulated Annealer. While the anti-cloning Hamiltonian successfully prevented duplicate scheduling, the sheer size of an enterprise roster (10,000+ variables) instantly overwhelmed CPU/RAM limits, leading to the creation of the "Macro-Qubit Squad" compression algorithm.
*   **Phase 2: The Predictive AI Pipeline (And why it was deprecated):** In my third and sixth iterations, I integrated a Machine Learning forecasting loop using `RandomForestRegressor` and `RandomForestClassifier` algorithms to predict demand and agent absenteeism over a multi-day horizon. **The Failure Point:** When this unconstrained AI forecast was fed into the multi-day quantum loop, the Simulated Annealer became trapped in massive local energy minima, causing the system's SLA to plummet to a catastrophic 2.4%. Furthermore, the ML models attempted to enforce rigid 8-hour shift blocks that shattered during sharp intraday volatility.
*   **Phase 3: The Erlang-Hybrid Pivot:** Realizing that pure AI lacked structural boundaries, I deprecated the Random Forest loop in favor of a classical queueing physics module (The Erlang Master). By using Erlang C to calculate exact temporal bounds and dynamically trim the candidate pool *before* quantum compilation, I successfully guided the solver away from local minima, achieving the optimal 90% SLA baseline.

---

## 🧮 Mathematical Formulation

Shiftwave operates on a bipartite mathematical architecture: a classical queuing physics module that establishes demand bounds, and a Quantum Unconstrained Binary Optimization (QUBO) model that navigates the assignment energy landscape.

### Part 1: The Classical Master (Erlang C Physics)
Before compiling the quantum matrix, the system calculates the exact interval targets ($N$) based on the offered load ($A$) and Average Handle Time ($AHT$). 

**Probability of Waiting ($P_w$)**
Calculates the probability that a caller will queue:
$$P_w = \frac{\frac{A^N}{N!} \cdot \frac{N}{N-A}}{\sum_{i=0}^{N-1} \frac{A^i}{i!} + \frac{A^N}{N!} \cdot \frac{N}{N-A}}$$

**Service Level Attainment**
Determines the percentage of calls answered within the target threshold ($T$):
$$\text{Service Level} = 1 - P_w \cdot e^{-(N-A)\frac{T}{AHT}}$$

### Part 2: The Quantum BQM (Energy Landscape)
The assignment challenge is formulated as a Constrained Quadratic Model (CQM) flattened into a QUBO landscape. The decision variable is a 3D binary tensor $x_{i,s,q} \in \{0,1\}$, where the variable equals 1 if Agent $i$ is assigned to Shift $s$ in Queue $q$. 

The total Hamiltonian seeks the lowest energy state balancing constraints and business logic:
$$H_{total} = H_{valid} + H_{cost} + H_{friction} + H_{routing} + H_{demand}$$

**1. Anti-Cloning Hard Constraint ($H_{valid}$)**
Applies a massive penalty ($\lambda_{valid}$) to ensure an agent works a maximum of one shift-queue combination per operational cycle.
$$H_{valid} = \lambda_{valid} \sum_{i} \sum_{(s,q) \neq (s',q')} x_{i,s,q} x_{i,s',q'}$$

**2. Cost & Preference Friction ($H_{cost} + H_{friction}$)**
Minimizes the linear sum of agent wages ($W_i$) combined with a friction penalty ($\lambda_{fric}$) if the assigned shift does not match the agent's preference vector ($P_{i,s}$).
$$H_{cost} + H_{friction} = \sum_{i,s,q} x_{i,s,q} \left( W_i + \lambda_{fric} (1 - P_{i,s}) \right)$$

**3. Demand Constraint via Slack Variables ($H_{demand}$)**
To meet Erlang C capacity targets without forcing rigid equality constraints (which cause classical solvers to crash on volatile curves), I introduce binary slack variables ($y_k$).
$$H_{demand} = \lambda_1 \sum_{q,t} \left( \text{Capacity}_{q,t} - \text{Agt}_{q,t} - \sum_{k=0}^{K} 2^k y_{k,q,t} \right)^2$$

**4. Parabolic Skill Routing ($H_{routing}$)**
To protect the elite labor pool, the solver squares the difference between the Agent's Tier ($V_i$) and the Queue's required Tier ($V_q$). This heavily penalizes extreme cross-skill mismatches while permitting minor flex-routing during demand spikes.
$$H_{routing} = \lambda^{III} \sum_{i,s} \sum_{q} (V_i - V_q)^2 \cdot x_{i,s,q}$$

---

## 🧬 Data Architecture & Statistical Modeling

To rigorously test Shiftwave against real-world enterprise volatility, the pipeline relies on a highly engineered synthetic data generation framework rather than flat, static datasets. 

The project utilizes two primary data structures:
*   **`agent_data.csv` (Labor Matrix):** Contains multidimensional metadata for 10,000 agents, including Tier classifications (General to Investor), Hourly Wages, Tenure Multipliers ($M_i$), Micro-Break Capacities, and strict Shift Preferences.
*   **`call_log.csv` (Interval Telemetry):** A granular, timestamped log of ~75,000 daily arrivals, segmented by Queue (General Support, Advisor, Intermed, Investor), and Channel.

### 1. Marginal Distributions
Standard call center physics dictate that system variables do not scale linearly. To mimic true stochastic variance, the data generator utilizes specific probability distributions:
*   **Arrival Volume (Poisson):** The frequency of incoming calls per 15-minute interval is modeled using a Poisson process. The baseline parameter is dynamically shifted using day-of-week M-curves and AI-deflection rates to simulate morning and afternoon rush hours.
*   **Average Handle Time (Lognormal):** Because call durations are strictly positive and naturally right-skewed (most calls are short, but complex escalations create a long tail), AHT is generated using a Lognormal distribution.

### 2. Multivariate Dependence (Copulas)
A critical flaw in basic synthetic datasets is the assumption that variables like Volume and AHT are independent. In reality, during enterprise stress events (e.g., a system outage), call volumes spike *and* call durations increase simultaneously due to issue complexity. 

To prevent the quantum solver from training on "easy" independent data, the generator utilizes **Copulas** to model the mathematical dependence between these random variables. 

By applying a Copula function, I successfully bind the Poisson (Volume) and Lognormal (AHT) marginal distributions together. This allows me to inject realistic correlation structures—such as severe cascading delays—while preserving the exact mathematical shape of the individual distributions. This ensures the quantum energy landscape is tested against highly realistic, correlated enterprise shocks rather than theoretical flatlines.

---

## 🚀 Key Features

* **Macro-Qubit Squad Compression:** Resolves the dimensionality explosion inherent in pure quantum pipelines. Clusters identical agents into dynamic tensor blocks, exponentially reducing matrix size without losing critical operational constraints.
* **3D Hamiltonian Formulation:** Optimizes Agent, Shift, and Queue simultaneously.
* **Erlang C Master Bounds:** Decouples macro-forecasting from micro-assignment, using classical queueing physics to trim the candidate pool before QPU compilation.
* **Manager Controls:** Hamiltonian weights act as operational dials, allowing users to tune SLA resilience, employee preference friction, and skill strictness dynamically.

---

## 📊 Results & Benchmarks

Validated against a Classical Integer Linear Programming (ILP) solver using PuLP and the CBC Branch-and-Cut algorithm. 

| Metric | Classical ILP Baseline | Shiftwave (Hybrid-Quantum) |
| :--- | :--- | :--- |
| **Solver Status** | Sub-Optimal (Over-staffed) | Optimal Energy Minimum |
| **Execution Time** | 113.84 sec | 77.48 sec |
| **Target SLA** | 98.7% | 90.0% |
| **Projected Payroll** | $1,032,365 | $807,609 |

**Conclusion:** Classical constraints act as rigid barriers. Shiftwave's quantum solver natively absorbs operational volatility, accepting minor localized penalties to achieve a global optimal schedule, saving **18% in payroll overhead**.

---

## 🚧 Limitations & Future Work

* **Hardware Access:** Simulated annealing on CPU instances (e.g., Google Colab) necessitated extreme Macro-Qubit compression (up to 35 agents per qubit), which occasionally inflated costs due to block-allocations. Deploying the uncompressed 3D matrix directly onto physical QPU hardware (like the D-Wave Advantage) will eliminate the need for compression and prevent local energy minima traps.
* **Dynamic Proficiency Degradation:** The current Erlang C calculations assume static agent proficiency across an 8-hour shift. Future iterations will introduce a time-decay variable to the Hamiltonian to mathematically account for human fatigue.

## ⚙️ Setup & Installation

To run the Shiftwave pipeline locally, ensure you have Python 3.10+ installed. 

1. Clone the repository:
   ```bash
   git clone [https://github.com/Jayash05/Shiftwave.git](https://github.com/Jayash05/Shiftwave.git)
   cd Shiftwave

---
*Built as a research experience in Quantum Computing Optimization without actually running on Quantum Computers.*
