"""
WorldSim — Multi-Agent World Environment
=========================================
Rewritten to exactly mirror the frontend simulation engine (frontend__1_.html).

Key design decisions:
  • 5 actions per agent:  0=CONSERVE  1=TRADE  2=EXPAND  3=RAID  4=ALLY
  • Resources kept in 0-100 range (same as frontend)
  • Q-table: 4 states × 5 actions  (identical to frontend)
  • Trust dict between regions  (-1 → +1)
  • Alliance list  [{a, b, formed}]
  • Collapse / Regrowth mechanics  (identical thresholds)
  • Climate events  (identical event table)
  • econPts + econDelta scoring  (identical FACTORS)
  • Frontend-compatible state payload broadcasted via WebSocket

The `get_frontend_state()` method returns a dict that maps
1-to-1 onto the frontend's `S` object so the UI can consume
it directly without any translation layer.
"""

import math
import random
from typing import Dict, List, Optional, Tuple

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────

NUM_REGIONS   = 5
MAX_STEPS     = 500

# Action indices — same as frontend ACTS[]
ACT_CONSERVE = 0
ACT_TRADE    = 1
ACT_EXPAND   = 2
ACT_RAID     = 3
ACT_ALLY     = 4
TOTAL_ACTIONS = 5

ACTS = ["CONSERVE", "TRADE", "EXPAND", "RAID", "ALLY"]
AICO = ["🌱", "🤝", "🏰", "⚔", "🛡"]

# ─────────────────────────────────────────────
# REGION DEFINITIONS
# Aligned with the frontend REALM_DEFS but keeping
# the real-world geography for distance calculation.
# Resources are init'd in 0-100 range.
# ─────────────────────────────────────────────

REGION_DEFS = [
    {
        "id": 0, "name": "THE NORTH",    "short": "NORTH",
        "color": "#7ab0e8",
        "coords": (65.0, 25.0),          # Northern Europe
        "x": 0.18, "y": 0.18,
        "r": {"water": 80, "food": 40, "energy": 50, "land": 75},
    },
    {
        "id": 1, "name": "WESTERLANDS",  "short": "WEST",
        "color": "#d07840",
        "coords": (25.0, -10.0),         # Western Europe / Iberia
        "x": 0.20, "y": 0.66,
        "r": {"water": 45, "food": 58, "energy": 88, "land": 62},
    },
    {
        "id": 2, "name": "VALYRIA",      "short": "VAL",
        "color": "#b870e0",
        "coords": (41.0, 29.0),          # Mediterranean / Anatolia
        "x": 0.55, "y": 0.22,
        "r": {"water": 52, "food": 48, "energy": 82, "land": 42},
    },
    {
        "id": 3, "name": "RIVERLANDS",   "short": "RIVER",
        "color": "#58d068",
        "coords": (51.0, 30.0),          # Eastern Europe / river basin
        "x": 0.62, "y": 0.66,
        "r": {"water": 88, "food": 78, "energy": 42, "land": 72},
    },
    {
        "id": 4, "name": "IRON ISLANDS", "short": "IRON",
        "color": "#7888b0",
        "coords": (64.0, -22.0),         # North Atlantic islands
        "x": 0.82, "y": 0.40,
        "r": {"water": 38, "food": 62, "energy": 55, "land": 32},
    },
]

# ─────────────────────────────────────────────
# DISTANCE MATRIX  (haversine, km)
# ─────────────────────────────────────────────

def _haversine(c1: Tuple[float, float], c2: Tuple[float, float]) -> float:
    R = 6371.0
    lat1, lon1 = math.radians(c1[0]), math.radians(c1[1])
    lat2, lon2 = math.radians(c2[0]), math.radians(c2[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return R * 2 * math.asin(math.sqrt(a))

DISTANCE_MATRIX: List[List[float]] = [
    [_haversine(REGION_DEFS[i]["coords"], REGION_DEFS[j]["coords"]) for j in range(NUM_REGIONS)]
    for i in range(NUM_REGIONS)
]

# ─────────────────────────────────────────────
# FACTOR SCORING  (mirrors frontend FACTORS[])
# ─────────────────────────────────────────────

def _factor_pts(r: "RegionState") -> float:
    """Compute economy delta for one region — identical to frontend computeDelta()."""
    pts = 0.0
    res = r.res

    # food surplus
    v = res["food"] - 40
    pts += 15 if v > 20 else 8 if v > 0 else -4 if v > -15 else -14
    # water access
    v = res["water"]
    pts += 12 if v > 70 else 6 if v > 40 else -3 if v > 20 else -12
    # energy output
    v = res["energy"]
    pts += 14 if v > 75 else 7 if v > 50 else 0 if v > 25 else -8
    # land control
    v = res["land"]
    pts += 10 if v > 70 else 5 if v > 45 else 0 if v > 20 else -6
    # trade bonus
    pts += min(r.trade_count * 6, 40)
    # alliance bonus
    pts += r.alliance_count * 18
    # conflict penalty
    pts -= r.conflict_count * 6
    # stability
    stab = min(100, max(0, (res["food"] + res["water"]) / 2 - r.conflict_count * 4))
    pts += 12 if stab > 70 else 6 if stab > 45 else 0 if stab > 20 else -8
    # population growth
    pts += int((r.population - 50) / 10) * 4
    # climate hits
    pts += (-r.climate_hits) * 4
    # conservation
    pts += r.conserve_count * 3
    # expansion
    pts += r.expand_count * 5

    return pts


# ─────────────────────────────────────────────
# REGION STATE
# ─────────────────────────────────────────────

class RegionState:
    """
    Mirrors the frontend mkRegion() object exactly.
    All fields have the same names used in the frontend's S.regions[i].
    """

    def __init__(self, defn: dict):
        d = defn
        self.id    = d["id"]
        self.name  = d["name"]
        self.short = d["short"]
        self.color = d["color"]
        self.x     = d["x"]
        self.y     = d["y"]

        # Resources: 0-100 scale
        self.res: Dict[str, float]      = {k: float(v) for k, v in d["r"].items()}
        self.prev_res: Dict[str, float] = {k: float(v) for k, v in d["r"].items()}

        # Population (55-85 range, same as frontend)
        self.population: int = 55 + random.randint(0, 29)

        # Phase: thriving / stable / struggling / critical / collapsed / regrowth
        self.phase: str  = "stable"
        self.alive: bool = True
        self.age:   int  = 0

        # Collapse / regrowth tracking
        self.collapse_count:      int  = 0
        self.regrowth_count:      int  = 0
        self.collapse_cycle:      int  = 0
        self.regrowth_cycle:      int  = 0
        self.regrowth_scheduled:  Optional[int] = None
        self.collapse_memory: dict = {"q": None, "lastAction": None, "cause": None}

        # Economy scoring
        self.econ_pts:   float = 100.0
        self.econ_delta: float = 0.0

        # Q-table: 4 resource-states × 5 actions
        # Initial values mirror frontend: [0.15, 0.30, 0.20, 0.10, 0.18] + small noise
        self.q: List[List[float]] = [
            [base + random.uniform(0, 0.06) for base in [0.15, 0.30, 0.20, 0.10, 0.18]]
            for _ in range(4)
        ]
        self.epsilon:    float = 0.45

        # RL memory
        self.last_action: Optional[int] = None
        self.last_reward: float = 0.0

        # Per-cycle counters (reset each step)
        self.trade_count:   int = 0
        self.alliance_count: int = 0
        self.conflict_count: int = 0
        self.conserve_count: int = 0
        self.expand_count:  int = 0
        self.climate_hits:  int = 0

        # Cumulative counters
        self.total_trades:    int = 0
        self.total_alliances: int = 0
        self.total_conflicts: int = 0

        # Trust dict: other_id → float (-1 to +1)
        self.trust: Dict[int, float] = {}

    # ── helpers ──

    def resource_state(self) -> int:
        """Map avg resource to Q-table row index (0-3). Same as frontend rsOf()."""
        avg = (self.res["water"] + self.res["food"] + self.res["energy"] + self.res["land"]) / 4
        if avg < 20: return 0
        if avg < 40: return 1
        if avg < 65: return 2
        return 3

    def best_resource(self) -> str:
        return max(self.res, key=lambda k: self.res[k])

    def worst_resource(self) -> str:
        return min(self.res, key=lambda k: self.res[k])

    def pick_action(self) -> Optional[int]:
        """ε-greedy action selection. Mirrors frontend pickAction()."""
        if not self.alive:
            return None
        # Regrowth bias: strongly favour CONSERVE / TRADE
        if self.phase == "regrowth":
            if random.random() < 0.5: return ACT_CONSERVE
            if random.random() < 0.5: return ACT_TRADE
        if random.random() < self.epsilon:
            return random.randint(0, 4)
        row = self.q[self.resource_state()]
        return row.index(max(row))

    def update_q(self, state: int, action: int, reward: float, next_state: int):
        """Q-learning update. Mirrors frontend updQ()."""
        alpha, gamma = 0.18, 0.88
        best_next = max(self.q[next_state])
        self.q[state][action] += alpha * (reward + gamma * best_next - self.q[state][action])
        self.epsilon = max(0.04, self.epsilon * 0.9985)

    def compute_phase(self, cycle: int) -> str:
        """Mirrors frontend computePhase()."""
        if not self.alive:
            return "collapsed"
        avg = (self.res["water"] + self.res["food"] + self.res["energy"] + self.res["land"]) / 4
        if self.phase == "regrowth" and (cycle - self.regrowth_cycle) < 20:
            return "regrowth"
        if avg > 65: return "thriving"
        if avg > 45: return "stable"
        if avg > 28: return "struggling"
        return "critical"

    def natural_depletion(self):
        """Mirrors frontend naturalDepletion()."""
        self.prev_res = dict(self.res)
        if self.phase == "regrowth":
            pf = 0.5
        elif self.phase == "struggling":
            pf = 0.8
        else:
            pf = self.population / 100.0

        def C(v, lo=0.0, hi=100.0): return min(hi, max(lo, v))

        self.res["water"]  = C(self.res["water"]  - (1.1 + random.random() * 0.9) * pf)
        self.res["food"]   = C(self.res["food"]   - (1.4 + random.random() * 1.4) * pf)
        self.res["energy"] = C(self.res["energy"] - (0.8 + random.random() * 0.9) * pf)
        self.res["land"]   = C(self.res["land"]   - 0.2 * random.random())

    def to_frontend_dict(self) -> dict:
        """
        Serialise to the exact shape the frontend uses for S.regions[i].
        Field names mirror the JS object properties.
        """
        return {
            "id":           self.id,
            "name":         self.name,
            "short":        self.short,
            "color":        self.color,
            "x":            self.x,
            "y":            self.y,
            "res":          {k: round(v, 2) for k, v in self.res.items()},
            "prevRes":      {k: round(v, 2) for k, v in self.prev_res.items()},
            "population":   self.population,
            "phase":        self.phase,
            "alive":        self.alive,
            "age":          self.age,
            "collapseCount":    self.collapse_count,
            "regrowthCount":    self.regrowth_count,
            "collapseCycle":    self.collapse_cycle,
            "regrowthCycle":    self.regrowth_cycle,
            "_regrowthScheduled": self.regrowth_scheduled,
            "collapseMemory":   self.collapse_memory,
            "econPts":      round(self.econ_pts, 2),
            "econDelta":    round(self.econ_delta, 2),
            "q":            [[round(v, 4) for v in row] for row in self.q],
            "epsilon":      round(self.epsilon, 4),
            "lastAction":   self.last_action,
            "lastReward":   round(self.last_reward, 4),
            "tradeCount":   self.trade_count,
            "allianceCount": self.alliance_count,
            "conflictCount": self.conflict_count,
            "conserveCount": self.conserve_count,
            "expandCount":  self.expand_count,
            "climateHits":  self.climate_hits,
            "totalTrades":  self.total_trades,
            "totalAlliances": self.total_alliances,
            "totalConflicts": self.total_conflicts,
            "trust":        {str(k): round(v, 4) for k, v in self.trust.items()},
            # animation helper — frontend sets this itself but we include it for completeness
            "_prevPhase":   self.phase,
        }


# ─────────────────────────────────────────────
# WORLD ENVIRONMENT
# ─────────────────────────────────────────────

class WorldEnv:
    """
    Mirrors the frontend simulation engine exactly.
    Run step() each cycle; get_frontend_state() returns the full S payload.
    """

    def __init__(self):
        self.step_count: int = 0
        self.regions:    List[RegionState] = []
        self.alliances:  List[dict] = []        # [{a, b, formed}]
        self.recent_trades: List[dict] = []     # [{from, to, res, amt, cycle}]
        self.climate_events: List[str] = []
        self.emergent_behaviors: List[str] = []
        self.econ_history: List[dict] = []      # [{cycle, pts:[...]}]
        self.event_log: List[str] = []          # raw event strings for WS
        self.total_trades:    int = 0
        self.total_conflicts: int = 0
        self.total_alliances: int = 0
        self.total_collapses: int = 0
        self.total_regrowths: int = 0
        self.leader_short:    str = "—"
        self._reset_regions()

    # ── public API ──

    def reset(self):
        self.step_count      = 0
        self.alliances       = []
        self.recent_trades   = []
        self.climate_events  = []
        self.emergent_behaviors = []
        self.econ_history    = []
        self.event_log       = []
        self.total_trades    = 0
        self.total_conflicts = 0
        self.total_alliances = 0
        self.total_collapses = 0
        self.total_regrowths = 0
        self.leader_short    = "—"
        self._reset_regions()

    def step(self) -> dict:
        """
        Run one simulation cycle. Returns a frontend-compatible event payload.
        Mirrors frontend simStep() exactly.
        """
        self.step_count += 1
        cycle = self.step_count

        # Reset per-cycle counters
        for r in self.regions:
            r.trade_count = r.alliance_count = r.conflict_count = 0
            r.conserve_count = r.expand_count = r.climate_hits = 0

        events = []   # log lines produced this step

        # Climate
        climate_event = self._do_climate(cycle)
        if climate_event:
            events.append(climate_event)

        # Regrowth
        regrowth_events = self._handle_regrowth(cycle)
        events.extend(regrowth_events)

        # Each region acts
        for r in self.regions:
            if not r.alive:
                continue
            r.natural_depletion()
            s = r.resource_state()
            action = r.pick_action()
            if action is None:
                continue
            reward, act_events = self._exec_action(r, action, cycle)
            ns = r.resource_state()
            r.update_q(s, action, reward, ns)
            r.last_action = action
            r.last_reward = reward
            r.age += 1
            events.extend(act_events)
            r.phase = r.compute_phase(cycle)

        # Collapse check
        collapse_events = self._check_collapse(cycle)
        events.extend(collapse_events)

        # Emergent behaviour detection
        self._detect_emergent(cycle)

        # Economy scoring
        max_pts = -999999
        leader = None
        for r in self.regions:
            delta = _factor_pts(r) if r.alive else -2.0
            r.econ_delta = delta
            r.econ_pts = max(10.0 if r.alive else 0.0, r.econ_pts + delta)
            if r.alive and r.econ_pts > max_pts:
                max_pts = r.econ_pts
                leader = r

        if leader:
            self.leader_short = leader.short

        self.econ_history.append({
            "cycle": cycle,
            "pts": [round(r.econ_pts, 2) for r in self.regions],
        })
        if len(self.econ_history) > 300:
            self.econ_history.pop(0)

        self.event_log.extend(events)
        if len(self.event_log) > 200:
            self.event_log = self.event_log[-200:]

        return {
            "type":   "world_state",
            "cycle":  cycle,
            "events": events,
        }

    def get_frontend_state(self) -> dict:
        """
        Returns the complete S object the frontend expects.
        Can be used to initialise or sync the frontend state.
        """
        return {
            "type":             "init",
            "cycle":            self.step_count,
            "running":          False,   # frontend controls this flag
            "totalTrades":      self.total_trades,
            "totalConflicts":   self.total_conflicts,
            "totalAlliances":   self.total_alliances,
            "totalCollapses":   self.total_collapses,
            "totalRegrowths":   self.total_regrowths,
            "climateEvents":    self.climate_events,
            "emergentBehaviors": self.emergent_behaviors,
            "econHistory":      self.econ_history,
            "recentTrades":     self.recent_trades,
            "alliances":        self.alliances,
            "regions":          [r.to_frontend_dict() for r in self.regions],
            "leaderShort":      self.leader_short,
        }

    # ── internal simulation ──

    def _reset_regions(self):
        self.regions = [RegionState(d) for d in REGION_DEFS]
        # Initialise trust between all pairs
        for r in self.regions:
            for o in self.regions:
                if r.id != o.id:
                    r.trust[o.id] = random.uniform(-0.1, 0.3)

    @staticmethod
    def _C(v: float, lo: float = 0.0, hi: float = 100.0) -> float:
        return min(hi, max(lo, v))

    # ── action execution ──

    def _exec_action(self, r: RegionState, action: int, cycle: int) -> Tuple[float, List[str]]:
        """
        Mirrors frontend execAction() exactly, including reward values
        and log message formats the frontend event log expects.
        """
        C = self._C
        reward = 0.0
        logs: List[str] = []
        alive_others = [x for x in self.regions if x.alive and x.id != r.id]

        # ── CONSERVE ──
        if action == ACT_CONSERVE:
            r.res["water"]  = C(r.res["water"]  + 3 + random.random() * 3)
            r.res["food"]   = C(r.res["food"]   + 3 + random.random() * 4)
            r.res["energy"] = C(r.res["energy"] + 2 + random.random() * 3)
            reward = 4 + random.random() * 2
            r.conserve_count += 1
            logs.append(f"{r.short} conserves resources — reserves stabilized")

        # ── TRADE ──
        elif action == ACT_TRADE:
            if alive_others:
                partner = max(alive_others, key=lambda o: r.trust.get(o.id, 0))
                give = r.best_resource()
                get  = r.worst_resource()
                amt  = 10 + random.randint(0, 11)
                if r.res[give] > 28:
                    r.res[give]        = C(r.res[give]        - amt * 0.65)
                    partner.res[give]  = C(partner.res[give]  + amt * 0.50)
                    r.res[get]         = C(r.res[get]         + amt * 0.55)
                    partner.res[get]   = C(partner.res[get]   - amt * 0.30)
                    r.trust[partner.id]        = C(r.trust.get(partner.id, 0) + 0.07, -1, 1)
                    partner.trust[r.id]        = C(partner.trust.get(r.id, 0) + 0.07, -1, 1)
                    r.trade_count += 1
                    r.total_trades += 1
                    self.total_trades += 1
                    reward = 7 + random.random() * 3
                    self.recent_trades.insert(0, {
                        "from": r.short, "to": partner.short,
                        "res": give, "amt": round(amt), "cycle": cycle,
                    })
                    if len(self.recent_trades) > 10:
                        self.recent_trades.pop()
                    logs.append(f"{r.name} ↔ {partner.name}: {give}×{round(amt)} traded")
                else:
                    reward = 1.0

        # ── EXPAND ──
        elif action == ACT_EXPAND:
            cost = {"food": 4, "water": 3, "energy": 3, "land": 2}
            can  = all(r.res[k] >= v + 12 for k, v in cost.items())
            if can:
                for k, v in cost.items():
                    r.res[k] = C(r.res[k] - v)
                r.res["land"] = C(r.res["land"] + 5)
                r.population  = min(200, r.population + 4)
                reward = 5.0
                r.expand_count += 1
                logs.append(f"{r.name} expands (+pop +land)")
            else:
                reward = -1.0

        # ── RAID ──
        elif action == ACT_RAID:
            targets = [o for o in alive_others if o.res["food"] > 25 or o.res["energy"] > 25]
            if targets:
                t = min(targets, key=lambda o: o.res["food"] + o.res["water"] + o.res["energy"])
                self.total_conflicts += 1
                r.conflict_count += 1
                r.total_conflicts += 1
                t.conflict_count += 1
                r.trust[t.id]  = C(r.trust.get(t.id, 0) - 0.20, -1, 1)
                t.trust[r.id]  = C(t.trust.get(r.id, 0) - 0.35, -1, 1)
                # Break any existing alliance
                before = len(self.alliances)
                self.alliances = [
                    a for a in self.alliances
                    if not ((a["a"] == r.id and a["b"] == t.id) or
                            (a["a"] == t.id and a["b"] == r.id))
                ]
                if len(self.alliances) < before:
                    r.econ_pts -= 12
                    t.econ_pts -= 12
                    logs.append(f"⚔ Alliance {r.short}↔{t.short} SHATTERED")
                # Combat resolution
                if random.random() < 0.5:
                    res_key = r.worst_resource()
                    loot = 10 + random.randint(0, 11)
                    t.res[res_key] = C(t.res[res_key] - loot * 0.65)
                    r.res[res_key] = C(r.res[res_key] + loot * 0.60)
                    reward = 5.0
                    logs.append(f"⚔ {r.short} raids {t.short} — seized {loot} {res_key}")
                else:
                    r.res["energy"] = C(r.res["energy"] - 8)
                    r.res["food"]   = C(r.res["food"] - 5)
                    reward = -4.0
                    logs.append(f"⚔ {r.short} REPELLED by {t.short}")

        # ── ALLY ──
        elif action == ACT_ALLY:
            candidates = [
                o for o in alive_others
                if r.trust.get(o.id, 0) > 0.22
                and not any(
                    (a["a"] == r.id and a["b"] == o.id) or
                    (a["a"] == o.id and a["b"] == r.id)
                    for a in self.alliances
                )
            ]
            if candidates:
                p = max(candidates, key=lambda o: r.trust.get(o.id, 0))
                self.alliances.append({"a": r.id, "b": p.id, "formed": cycle})
                r.alliance_count  += 1
                r.total_alliances += 1
                p.alliance_count  += 1
                self.total_alliances += 1
                r.trust[p.id] = C(r.trust.get(p.id, 0) + 0.15, -1, 1)
                p.trust[r.id] = C(p.trust.get(r.id, 0) + 0.15, -1, 1)
                reward = 10.0
                logs.append(f"🛡 ALLIANCE: {r.name} ↔ {p.name}")
            else:
                reward = 0.0

        return reward, logs

    # ── climate ──

    def _do_climate(self, cycle: int) -> Optional[str]:
        """Mirrors frontend doClimate(). 16% chance per step."""
        if random.random() > 0.16:
            return None
        events = [
            {"n": "DROUGHT",    "res": "water",  "mod": -18, "bad": True},
            {"n": "FAMINE",     "res": "food",   "mod": -20, "bad": True},
            {"n": "SOLAR FLARE","res": "energy", "mod": -16, "bad": True},
            {"n": "FLOOD",      "res": "land",   "mod": -12, "bad": True},
            {"n": "RAINFALL",   "res": "water",  "mod": +22, "bad": False},
            {"n": "HARVEST",    "res": "food",   "mod": +20, "bad": False},
            {"n": "GEO SURGE",  "res": "energy", "mod": +18, "bad": False},
        ]
        e = random.choice(events)
        alive = [r for r in self.regions if r.alive]
        if not alive:
            return None
        affected = random.sample(alive, min(len(alive), 1 + random.randint(0, 1)))
        names = []
        for r in affected:
            r.res[e["res"]] = self._C(r.res[e["res"]] + e["mod"])
            if e["bad"]:
                r.climate_hits += 1
            names.append(r.short)
        entry = f"[{str(cycle).zfill(3)}] {e['n']} → {', '.join(names)}"
        self.climate_events.insert(0, entry)
        if len(self.climate_events) > 15:
            self.climate_events.pop()
        msg = f"🌡 {e['n']} hits {', '.join(names)}"
        return msg

    # ── collapse ──

    def _check_collapse(self, cycle: int) -> List[str]:
        """Mirrors frontend checkAndHandleCollapse()."""
        logs = []
        for r in self.regions:
            if not r.alive:
                continue
            avg = (r.res["water"] + r.res["food"] + r.res["energy"] + r.res["land"]) / 4
            if avg >= 8:
                continue
            # COLLAPSE
            r.alive   = False
            r.phase   = "collapsed"
            r.collapse_count += 1
            r.collapse_cycle  = cycle
            self.total_collapses += 1
            r.econ_pts = max(0.0, r.econ_pts - 40)
            cause = self._identify_cause(r)
            r.collapse_memory = {
                "q":          [[round(v, 4) for v in row] for row in r.q],
                "lastAction": r.last_action,
                "cause":      cause,
            }
            # Q-table punishment — same as frontend
            if r.last_action is not None:
                r.q[0][r.last_action] -= 0.3
                r.q[1][r.last_action] -= 0.2
            r.q[0][ACT_CONSERVE] += 0.4
            r.q[0][ACT_TRADE]    += 0.3
            r.q[1][ACT_CONSERVE] += 0.2
            r.q[1][ACT_TRADE]    += 0.2
            # Remove from alliances
            self.alliances = [a for a in self.alliances if a["a"] != r.id and a["b"] != r.id]
            r.regrowth_scheduled = cycle + 15
            logs.append(f"💀 {r.name.upper()} COLLAPSED — {cause}")
            self._add_behavior(f"{r.short} collapsed ({cause}) — Q-table punished")
        return logs

    def _identify_cause(self, r: RegionState) -> str:
        low = [k for k in ["water", "food", "energy", "land"] if r.res[k] < 15]
        if low:
            return " + ".join(low) + " depletion"
        best_a = r.q[2].index(max(r.q[2]))
        return f"over-reliance on {ACTS[best_a]}"

    # ── regrowth ──

    def _handle_regrowth(self, cycle: int) -> List[str]:
        """Mirrors frontend handleRegrowth()."""
        logs = []
        for r in self.regions:
            if r.alive:
                continue
            if r.regrowth_scheduled is None or cycle < r.regrowth_scheduled:
                continue
            r.alive    = True
            r.phase    = "regrowth"
            r.regrowth_count += 1
            r.regrowth_cycle  = cycle
            self.total_regrowths += 1
            r.regrowth_scheduled = None
            r.res["water"]  = 20 + random.random() * 15
            r.res["food"]   = 18 + random.random() * 15
            r.res["energy"] = 18 + random.random() * 15
            r.res["land"]   = 20 + random.random() * 15
            r.population    = max(30, r.population - 20)
            r.econ_pts     += 20
            r.epsilon       = min(0.35, r.epsilon + 0.2)
            logs.append(f"🌱 {r.name.upper()} REGROWS — {r.collapse_count}x collapse, learning applied")
            self._add_behavior(f"{r.short} achieved regrowth — applying learned penalties")
        return logs

    # ── emergent behaviour detection ──

    def _detect_emergent(self, cycle: int):
        """Mirrors frontend detectEmergent(). Fires every 20 cycles."""
        if cycle % 20 != 0:
            return
        for r in self.regions:
            if not r.alive:
                continue
            best = r.q[2].index(max(r.q[2]))
            if best == ACT_TRADE and r.total_trades > 8:
                self._add_behavior(f"{r.short} emerged as TRADE hub ({r.total_trades} trades)")
            if best == ACT_CONSERVE and r.res["water"] < 35:
                self._add_behavior(f"{r.short} learned CONSERVATION after water crisis")
            if best == ACT_RAID and r.res["food"] > 65 and r.collapse_count == 0:
                self._add_behavior(f"{r.short} uses AGGRESSION to guard surplus")
            if best == ACT_ALLY and r.total_alliances > 2:
                self._add_behavior(f"{r.short} mastered ALLIANCE diplomacy")
            if r.collapse_count > 0 and r.regrowth_count > 0 and r.phase != "collapsed":
                self._add_behavior(
                    f"{r.short} rebuilt ({r.collapse_count} collapse, {r.regrowth_count} regrowth)"
                )
        for r in self.regions:
            for o in self.regions:
                if r.id < o.id and r.trust.get(o.id, 0) > 0.55:
                    self._add_behavior(
                        f"{r.short}↔{o.short} deep trust ({int(r.trust[o.id]*100)}%)"
                    )

    def _add_behavior(self, b: str):
        if b not in self.emergent_behaviors:
            self.emergent_behaviors.append(b)
            if len(self.emergent_behaviors) > 12:
                self.emergent_behaviors.pop(0)

    # ── distance helper (for agents/train.py) ──

    @staticmethod
    def get_distance(i: int, j: int) -> float:
        return DISTANCE_MATRIX[i][j]
