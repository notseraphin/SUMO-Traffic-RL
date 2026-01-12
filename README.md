# SUMO-Traffic-RL
**Adaptive Traffic Signal Control using Reinforcement Learning**

***Overview***

This project implements a reinforcement learning (Q-learning) agent to adaptively control traffic lights on a 4×4 urban grid simulated using SUMO.

The objective is to optimize traffic flow by reducing vehicle halting and increasing throughput through adaptive traffic signal phases.

Key outcomes include:

- Dynamic traffic light phase selection based on real-time lane occupancy
- Measurable improvements in total vehicle throughput
- Reduced vehicle halting across simulation episodes
- Generalizable agent policies capable of handling variable traffic scenarios

The pipeline integrates SUMO simulations with Python-based Q-learning for end-to-end learning and evaluation.

***Data Source***

- SUMO-generated traffic network (4×4 grid)
- Vehicle flows defined in .rou.xml route files
- Network structure and traffic lights defined in .net.xml and .sumocfg

Derived datasets:

- q_table.pkl — learned Q-values for state-action pairs

***Model / Analysis Description***

**Environment Design**

- State representation: discretized vehicle counts per lane controlled by each traffic light
- Action space: traffic light phases for each signal
- Reward: negative sum of halting vehicles per timestep to incentivize smooth traffic flow

**Q-Learning Agent**

- ε-greedy policy for exploration
- Learning rate (α) and discount factor (γ) tuned for convergence
- Q-table maps state tuples to action tuples representing traffic light phases

**Training & Evaluation**

- Episodes: 25 simulation episodes
- Steps per episode: 1,000 simulation steps
- Performance metrics: total reward, cumulative vehicle halting, throughput

Project Structure
```bash
RL-Traffic-Signal-Control/
│── env/
│   └── traffic_env.py
│
│── mdp/
│   ├── agent.py    
│   └── q_table.pkl
│
│── sumo/
│   ├── network/
│   │   ├── grid.net.xml
│   │   └── grid.sumocfg
│   └── routes/
│       └── routes.rou.xml
│
│── experiments/
│   └── run_simulation.py
│
│── simulation_results.txt
│── README.md
│── requirements.txt
```

***lResults Interpretation***

- The agent progressively reduces vehicle halting across episodes
- Total episode reward generally improves with training, reflecting smoother traffic flow
- Learned policies generalize to dynamic traffic scenarios by selecting phase combinations that minimize congestion

***Insights***

- Multi-agent adaptive traffic control significantly improves throughput versus static timing strategies
- Network bottlenecks are mitigated when the agent coordinates signals based on lane occupancy
- Episodes highlight the impact of early exploration vs. later exploitation in learning optimal policies

***Overall Conclusion***

The project demonstrates the feasibility and effectiveness of reinforcement learning for urban traffic optimization:

- Q-learning can successfully learn adaptive traffic light policies in a simulated environment
- Simulation-based evaluation provides actionable metrics for traffic flow improvement
- The approach can be scaled to larger or more complex networks and integrated with real-time traffic data

***Technologies Used***

- Python
- NumPy, Pandas
- SUMO (Simulation of Urban Mobility)
- TraCI (Python API for SUMO)
- Q-Learning (Reinforcement Learning)
