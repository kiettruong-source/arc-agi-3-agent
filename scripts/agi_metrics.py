"""Additional metrics, computed from run files (protocol section 2, "additional metrics").

uv run python scripts/agi_metrics.py runs/<file>.json [--seed N] [--vs runs/<other>.json ...]

One number per capability the benchmark names (exploration, modeling, goal-setting, planning) plus efficiency,
breadth, transfer and cost. `--vs` puts other runs next to the first one and shows the difference:
  held-out run vs dev run      -> generalization gap
  --rename N run vs plain run  -> renaming robustness
  --fresh-memory run vs plain  -> memory ablation (cross-game transfer)
"""
import argparse
import json
import statistics

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--vs", nargs="*", default=[])
args = ap.parse_args()
med = lambda xs: statistics.median(xs) if xs else float("nan")


def load(path):
    r = json.load(open(path))
    run = r["per_seed"][min(args.seed, len(r["per_seed"]) - 1)]
    rows = []  # (game, level, human, ours, route)
    for g, v in run["games"].items():
        for i in range(v["levels"]):
            route = v["best_play_level_actions"][i] if i < (v.get("best_play_levels") or 0) else None
            rows.append((g, i + 1, v["human_level_actions"][i], v["level_actions"][i], route))
    games, use, pr = run["games"], run["skill_usage"], run.get("prediction") or {}
    known = sum(n for k, n in use.items() if k in ("replay", "go_to", "click_all", "repeat"))
    right = sum(n for k, n in pr.items() if k.endswith(" right"))
    wrong = sum(n for k, n in pr.items() if k.endswith(" wrong"))
    tested = [x for v in games.values() for x in v.get("goals", []) if x.get("goal_kind")]
    secs = sum(v.get("agent_seconds", 0) for v in games.values())
    acts = sum(v.get("played_actions", 0) for v in games.values())
    m = {
        "games": len(games),
        "levels solved (first exposure)": sum(v["levels"] for v in games.values()),
        "score (first exposure)": statistics.mean(v["score"] for v in games.values()),
        "score (best play)": statistics.mean(v["best_play_score"] for v in games.values()),
        "breadth: share of games with a level solved": sum(v["levels"] > 0 for v in games.values()) / len(games),
        "efficiency: solved levels within 5x the human count": sum(a <= 5 * h for _, _, h, a, _ in rows),
        "efficiency: actions relative to the human (median)": med([a / h for _, _, h, a, _ in rows]),
        "exploration: discovery cost, actions per route step (median)": med([a / rt for *_, a, rt in rows if rt]),
        "planning: route found / human count (median)": med([rt / h for _, _, h, _, rt in rows if rt]),
        "planning: share of actions from a known skill or plan": known / max(1, sum(use.values())),
        "modeling: share of actions with a prediction": (right + wrong) / pr["actions"] if pr.get("actions") else float("nan"),
        "modeling: predictions right": right / (right + wrong) if right + wrong else float("nan"),
        "modeling: commoner-outcome baseline": max(pr["changed"] / pr["actions"], 1 - pr["changed"] / pr["actions"]) if pr.get("actions") else float("nan"),
        "goal-setting: won levels that had a goal to test": len(tested),
        "goal-setting: of these, goal true on the winning screen": sum(bool(x["goal_true_at_win"]) for x in tested),
        "cost: agent ms per action": 1000 * secs / acts if acts else float("nan"),
        "cost: LLM questions": (r.get("llm") or {}).get("asked", 0),
    }
    return r, run, rows, m


r, run, rows, m = load(args.run)
print(f"{r['label']}  ({r['command']}, commit {r['commit']})\n")
print(f"{'game':5} {'level':>5} {'human':>6} {'ours':>6} {'ours/human':>10} {'within 5x':>9} {'route':>6} {'ours/route':>10} {'route/human':>11}")
for g, lv, h, a, rt in rows:
    print(f"{g:5} {lv:5} {h:6} {a:6} {a / h:10.1f} {'yes' if a <= 5 * h else 'no':>9} {rt or '-':>6} "
          f"{a / rt if rt else float('nan'):10.1f} {rt / h if rt else float('nan'):11.2f}")
for g, v in run["games"].items():
    if v["levels"] >= 2:
        ratios = [round(v["level_actions"][i] / v["human_level_actions"][i], 1) for i in range(v["levels"])]
        print(f"transfer inside {g}: actions relative to the human by level {ratios}   (falling = knowledge carried over)")

others = [load(p) for p in args.vs]
fmt = lambda x: f"{x:10.2f}" if isinstance(x, float) else f"{x:10}"
print("\n" + " " * 62 + f"{r['label'][:10]:>10}" + "".join(f"{o[0]['label'][:14]:>16}{'diff':>8}" for o in others))
for k, v in m.items():
    line = f"{k:62}{fmt(v)}"
    for o in others:
        line += f"{fmt(o[3][k]):>16}{o[3][k] - v:+8.2f}"
    print(line)
