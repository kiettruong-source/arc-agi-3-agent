"""Additional metrics, computed from a run file (protocol section 2, "additional metrics").

uv run python scripts/agi_metrics.py runs/<file>.json [seed]

One number per capability the benchmark names (exploration, planning) plus efficiency, breadth and transfer.
Only what the run record already holds; metrics that need new measurements are listed in the protocol.
"""
import json
import statistics
import sys

r = json.load(open(sys.argv[1]))
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
run = r["per_seed"][seed]
rows = []  # (game, level, human, ours, route)
for g, v in run["games"].items():
    for i in range(v["levels"]):
        route = v["best_play_level_actions"][i] if i < (v.get("best_play_levels") or 0) else None
        rows.append((g, i + 1, v["human_level_actions"][i], v["level_actions"][i], route))
med = lambda xs: round(statistics.median(xs), 2) if xs else None

print(f"{'game':5} {'level':>5} {'human':>6} {'ours':>6} {'ours/human':>10} {'within 5x':>9} {'route':>6} {'ours/route':>10} {'route/human':>11}")
for g, lv, h, a, rt in rows:
    print(f"{g:5} {lv:5} {h:6} {a:6} {a / h:10.1f} {'yes' if a <= 5 * h else 'no':>9} {rt or '-':>6} "
          f"{a / rt if rt else float('nan'):10.1f} {rt / h if rt else float('nan'):11.2f}")

games = run["games"]
use = run["skill_usage"]
exploit = sum(n for k, n in use.items() if k in ("replay", "go_to", "click_all", "repeat"))
print()
print(f"breadth: games with at least one level solved      {sum(v['levels'] > 0 for v in games.values())} of {len(games)}")
print(f"efficiency: solved levels within 5x the human count {sum(a <= 5 * h for _, _, h, a, _ in rows)} of {len(rows)}   (the official leaderboard stops a level at 5x)")
print(f"efficiency: actions relative to the human, median   {med([a / h for _, _, h, a, _ in rows])}x")
print(f"exploration: discovery cost (actions per step of the route found), median  {med([a / rt for *_, a, rt in rows if rt])}")
print(f"planning: route found vs human count, median        {med([rt / h for _, _, h, _, rt in rows if rt])}   (below 1 = shorter than the human baseline)")
print(f"planning: share of actions that execute a known skill or plan  {exploit / max(1, sum(use.values())):.0%}   (the rest is searching)")
two = [(g, v) for g, v in games.items() if v["levels"] >= 2]
for g, v in two:
    ratios = [v["level_actions"][i] / v["human_level_actions"][i] for i in range(v["levels"])]
    print(f"transfer inside {g}: actions relative to the human by level {[round(x, 1) for x in ratios]}   (falling = knowledge carried over)")
if not two:
    print("transfer inside a game: no game with two solved levels")
