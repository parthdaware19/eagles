# WorldSim — Adaptive Resource Scarcity Simulator

## Overview

WorldSim is a multi-region simulation engine where each region has limited resources:

- Water
- Food
- Energy
- Land

Each region is controlled by a Q-learning based AI agent.  
Agents learn survival, trade, expansion, and conflict strategies over time.

The world evolves through:

- Resource depletion
- Climate events
- Trade negotiations
- Conflicts
- Regional collapse

The system includes real-time visualization and analytics to study emergent strategies.

---

## Core Features

- Q-Learning agents
- Dynamic climate events (drought, flood, famine)
- Trade & diplomacy system
- Raid & conflict mechanism
- Collapse detection
- Live analytics dashboard
- Q-table visualization
- Interactive world map

---

## Simulation Cycle

Each cycle:

1. Climate event may occur  
2. Resources deplete  
3. Agents choose action:
   - Conserve
   - Trade
   - Expand
   - Raid  
4. Rewards calculated  
5. Q-table updated  
6. Trades/conflicts processed  
7. Collapse checked  
8. Data logged  

---

## Agent Model

### States
- Critical
- Low
- Medium
- High  
(Based on average resource level)

### Actions
- Conserve
- Trade
- Expand
- Raid

### Rewards
- Positive → sustainability, successful trade
- Negative → depletion, failed raid, collapse

Epsilon decreases over time for better policy convergence.

---

## Trade & Conflict

- Trade depends on surplus, trust score, and stability.
- Repeated successful trades build alliances.
- Failed trade reduces trust.
- Raid success depends on strength and stability.

---

## Emergent Analysis

Tracks:

- Dominant strategies
- Alliance formation
- Conflict frequency
- Collapse causes
- Resource sustainability

---

## Goal

To analyze:

- Which strategies survive long-term?
- Which collapse and why?
- What does this reveal about real-world resource conflicts?
