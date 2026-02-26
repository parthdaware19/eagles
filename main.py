"""
WorldSim — FastAPI + WebSocket Server
======================================
Broadcasts the exact frontend S-state each cycle so the UI
can consume it directly via WebSocket.

On startup:
  1. Loads trained Q-tables from models/qtables.json  (if present)
     so agents start with learned policies.
  2. Starts the background simulation loop.

WebSocket — /ws
  Server → client  every cycle:
    { type:"init" }         on first connect / after reset
    { type:"world_state" }  every simulation step
    { type:"reset" }        after reset command
  All payloads match the frontend S object exactly.

  Client → server  (optional — for external control):
    { type:"control", command:"play"|"pause"|"reset"|"speed", value?:float }

REST Endpoints
--------------
  GET  /               health check
  GET  /state          full S-state snapshot
  GET  /history        econHistory last N entries
  GET  /distance_matrix
  POST /control        play | pause | reset | speed
  POST /action         manually execute one action on one region
  POST /event          trigger a climate / disaster event

Run
---
  cd backend
  uvicorn server.main:app --reload --port 8000
"""

import asyncio, json, os, sys

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from env.world_env import WorldEnv, NUM_REGIONS, REGION_DEFS, ACTS, DISTANCE_MATRIX

# ══════════════════════════════════════════════════════════
# APP + CORS
# ══════════════════════════════════════════════════════════

app = FastAPI(title="WorldSim", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"],
                   allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ══════════════════════════════════════════════════════════
# GLOBAL STATE
# ══════════════════════════════════════════════════════════

env = WorldEnv()

sim = {"running": False, "speed": 1.0}   # steps-per-second

QTABLE_PATH = os.path.join(_ROOT, "models", "qtables.json")

# ══════════════════════════════════════════════════════════
# WEBSOCKET MANAGER
# ══════════════════════════════════════════════════════════

class WSManager:
    def __init__(self): self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept(); self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active: self.active.remove(ws)

    async def broadcast(self, data: dict):
        msg  = json.dumps(data)
        dead = []
        for ws in self.active:
            try: await ws.send_text(msg)
            except Exception: dead.append(ws)
        for ws in dead: self.disconnect(ws)

mgr = WSManager()

# ══════════════════════════════════════════════════════════
# SIMULATION LOOP
# ══════════════════════════════════════════════════════════

async def sim_loop():
    while True:
        if sim["running"]:
            events, summary = env.step()
            payload = env.get_frontend_state("world_state")
            payload["events"] = events
            await mgr.broadcast(payload)
            if env.step_count >= 500:
                env.reset()
                env.load_trained_qtables(QTABLE_PATH)
                rst = env.get_frontend_state("reset")
                await mgr.broadcast(rst)
        await asyncio.sleep(max(0.05, 1.0 / max(0.1, sim["speed"])))


@app.on_event("startup")
async def startup():
    # Inject trained Q-tables if available
    env.load_trained_qtables(QTABLE_PATH)
    asyncio.create_task(sim_loop())

# ══════════════════════════════════════════════════════════
# REST ENDPOINTS
# ══════════════════════════════════════════════════════════

@app.get("/")
def root():
    return {"status":"ok","step":env.step_count,"running":sim["running"],"speed":sim["speed"]}

@app.get("/state")
def get_state():
    return env.get_frontend_state("init")

@app.get("/history")
def get_history(last_n: int = 100):
    return {"econHistory": env.econ_history[-last_n:]}

@app.get("/distance_matrix")
def get_distances():
    return {
        "matrix": [[round(v,1) for v in row] for row in DISTANCE_MATRIX],
        "labels": [d["name"] for d in REGION_DEFS],
    }

# ── Control ───────────────────────────────────────────────

class ControlReq(BaseModel):
    command: str
    value:   Optional[float] = None

@app.post("/control")
async def control(req: ControlReq):
    cmd = req.command.lower()
    if   cmd == "play":  sim["running"] = True
    elif cmd == "pause": sim["running"] = False
    elif cmd == "reset":
        sim["running"] = False
        env.reset()
        env.load_trained_qtables(QTABLE_PATH)
        await mgr.broadcast(env.get_frontend_state("reset"))
    elif cmd == "speed":
        if not req.value or req.value <= 0:
            raise HTTPException(400, "speed requires positive float")
        sim["speed"] = float(req.value)
    else:
        raise HTTPException(400, f"Unknown command '{cmd}'")
    return {"status":"ok","sim":sim}

# ── Manual action ─────────────────────────────────────────

class ActionReq(BaseModel):
    region_id: int    # 0–4
    action:    int    # 0–4

@app.post("/action")
async def manual_action(req: ActionReq):
    if not 0 <= req.region_id < NUM_REGIONS:
        raise HTTPException(400, "region_id must be 0–4")
    if not 0 <= req.action < 5:
        raise HTTPException(400, f"action 0–4  ({', '.join(ACTS)})")
    r = env.regions[req.region_id]
    if not r.alive:
        raise HTTPException(400, f"{r.name} is collapsed")
    reward, events = env._exec_action(r, req.action, env.step_count)
    r.last_action = req.action; r.last_reward = reward
    payload = env.get_frontend_state("world_state")
    payload["events"] = events
    await mgr.broadcast(payload)
    return {"reward":round(reward,4),"events":events,"region":r.to_frontend_dict()}

# ── Event trigger ─────────────────────────────────────────

class EventReq(BaseModel):
    event_type: str   # drought|plague|boom|disaster|famine|solar_flare
    region_id:  int

@app.post("/event")
async def trigger_event(req: EventReq):
    if not 0 <= req.region_id < NUM_REGIONS:
        raise HTTPException(400, "region_id must be 0–4")
    r  = env.regions[req.region_id]
    et = req.event_type.lower().replace(" ","_")
    C  = lambda v: min(100.0,max(0.0,v))
    events = []
    if   et == "drought":     r.res["water"]=C(r.res["water"]*0.30); r.res["food"]=C(r.res["food"]*0.50); r.climate_hits+=1; events.append(f"🌵 DROUGHT struck {r.name}!")
    elif et == "plague":      r.population=max(30,int(r.population*0.75)); events.append(f"☠ PLAGUE hit {r.name}!")
    elif et == "boom":        [setattr(r,"res",{k:C(v*1.5) for k,v in r.res.items()})]; r.econ_pts+=30; events.append(f"💰 ECONOMIC BOOM in {r.name}!")
    elif et == "disaster":    r.population=max(30,int(r.population*0.85)); r.res["land"]=C(r.res["land"]*0.60); r.climate_hits+=2; events.append(f"🌋 DISASTER in {r.name}!")
    elif et == "famine":      r.res["food"]=C(r.res["food"]*0.20); r.res["water"]=C(r.res["water"]*0.60); r.climate_hits+=1; events.append(f"🌾 FAMINE ravages {r.name}!")
    elif et == "solar_flare": r.res["energy"]=C(r.res["energy"]*0.25); r.climate_hits+=1; events.append(f"☀ SOLAR FLARE disrupts {r.name}!")
    else: raise HTTPException(400, f"Unknown event '{et}'")
    collapse_logs = env._check_collapse(env.step_count)
    events.extend(collapse_logs)
    payload = env.get_frontend_state("world_state"); payload["events"]=events
    await mgr.broadcast(payload)
    return {"events":events,"region":r.to_frontend_dict()}

# ══════════════════════════════════════════════════════════
# WEBSOCKET
# ══════════════════════════════════════════════════════════

@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await mgr.connect(ws)
    try:
        await ws.send_text(json.dumps(env.get_frontend_state("init")))
        while True:
            data = await ws.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("type") == "control":
                    await control(ControlReq(command=msg.get("command",""), value=msg.get("value")))
            except Exception as e:
                await ws.send_text(json.dumps({"type":"error","detail":str(e)}))
    except WebSocketDisconnect:
        mgr.disconnect(ws)

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
