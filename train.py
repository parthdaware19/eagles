"""
WorldSim — Q-Learning Trainer
Runs the simulation headlessly for N episodes, lets agents
learn via Q-table updates, then saves qtables.json so
the server can load the trained policies on startup.

Run:
    cd backend/
    python -m agents.train              # 500 episodes
    python -m agents.train --episodes 2000
"""

import argparse, json, os, sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path: sys.path.insert(0, _ROOT)

from env.world_env import WorldEnv, NUM_REGIONS, ACTS, MAX_STEPS

SAVE_PATH = os.path.join(_ROOT, "agents", "models", "qtables.json")


def train(episodes: int = 500, verbose: bool = True):
    env = WorldEnv()
    best_total = -1e9

    for ep in range(1, episodes + 1):
        env.reset()
        ep_rewards = [0.0] * NUM_REGIONS

        for _ in range(MAX_STEPS):
            if all(not r.alive for r in env.regions):
                break
            events, summary = env.step()
            for i, r in enumerate(env.regions):
                ep_rewards[i] += r.last_reward

        total = sum(ep_rewards)
        if verbose and ep % 50 == 0:
            phases   = [r.phase[:4] for r in env.regions]
            epsilons = [f"{r.epsilon:.2f}" for r in env.regions]
            best_acts= [ACTS[r.q[2].index(max(r.q[2]))] for r in env.regions]
            print(
                f"  Ep {ep:4d}/{episodes}  "
                f"total_R={total:7.1f}  "
                f"collapses={env.total_collapses}  "
                f"trades={env.total_trades}  "
                f"phases={phases}  "
                f"ε={epsilons}"
            )
            if verbose and ep % 200 == 0:
                print(f"           best_acts={best_acts}")

        if total > best_total:
            best_total = total
            _save(env, ep)

    # Always save final
    _save(env, episodes, final=True)
    print(f"\n  Training complete. Best total reward: {best_total:.1f}")
    print(f"  Q-tables saved → {SAVE_PATH}")


def _save(env: WorldEnv, ep: int, final: bool = False):
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    data = {
        "episode": ep,
        "final":   final,
        "agents":  {
            str(r.id): {
                "name":    r.name,
                "q":       [[round(v, 5) for v in row] for row in r.q],
                "epsilon": round(r.epsilon, 5),
            }
            for r in env.regions
        }
    }
    with open(SAVE_PATH, "w") as f:
        json.dump(data, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int,  default=500)
    parser.add_argument("--quiet",    action="store_true")
    args = parser.parse_args()
    print(f"\n  WorldSim Q-Learning Trainer")
    print(f"  Episodes : {args.episodes}")
    print(f"  Save to  : {SAVE_PATH}")
    print(f"  {'='*50}")
    train(episodes=args.episodes, verbose=not args.quiet)
