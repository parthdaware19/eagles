# eagles
WORLDSIM
Technical Overview
WorldSim is a multi-agent reinforcement learning simulation that models resource-constrained geopolitical dynamics. Each region operates as an autonomous AI agent that learns optimal strategies for survival, trade, alliance formation, expansion, and conflict.

Core Architecture
Environment Layer

Finite resources: water, food, energy, land

Climate events and natural depletion

Trust-based diplomacy network

Agent Layer

State space: Critical, Low, Medium, High

Action space: Conserve, Trade, Expand, Raid, Ally

Q-learning with epsilon-greedy exploration

Adaptive trust matrix for diplomacy

Visualization Layer

Real-time world map (Canvas API)

Trade and alliance graph rendering

Economic ranking dashboard

Q-table insight panel

Reinforcement Learning Model
Q-update rule:

Q(s,a) ← Q(s,a) + α [ r + γ max Q(s’,a’) − Q(s,a) ]

α = 0.15 (learning rate)

γ = 0.88 (discount factor)

ε decays over time for convergence

Agents are not pre-programmed — strategies emerge through reward-driven learning.

Reward Engineering
Positive incentives:

Trade success

Alliance stability

Resource sustainability

Growth

Negative incentives:

Conflict

Trust loss

Resource collapse

Climate damage

Economic score aggregates these factors each cycle.

Emergent Outcomes
Trade dominance under resource complementarity

Aggressive strategies decline due to trust penalties

Alliances stabilize long-term growth

Survival mode shifts policy toward conservation

Technical Value
WorldSim demonstrates:

Multi-agent RL in dynamic environments

Incentive-driven equilibrium formation

Trust-network evolution

Real-time interactive simulation

A scalable prototype bridging reinforcement learning, economics, and geopolitical modeling.
