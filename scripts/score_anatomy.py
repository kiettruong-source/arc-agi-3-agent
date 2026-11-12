"""Where do the score points come from, and where are they lost?  (protocol E14)

uv run python scripts/score_anatomy.py runs/<file>.json [seed]

Score of a game = sum(level weight x level score) / sum(weights); level score = min((human / ours)^2, 1.15) x 100,
weight = level number. One "point" below = one unit of a game's 0-100 score; the overall score is the mean over games.
"""
import json
import sys

r = json.load(open(sys.argv[1]))
seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
games = r["per_seed"][seed]["games"]
tot = {"first": 0.0, "best": 0.0, "human": 0.0}
print(f"{'game':5} {'level':>5} {'human':>6} {'ours':>6} {'points':>7} | {'best play':>9} {'points':>7} | {'at human speed':>14}")
for g, v in games.items():
    n = v["level_count"]
    sw = n * (n + 1) / 2
    tot["first"] += v["score"]
    tot["best"] += v["best_play_score"]
    for i in range(max(v["levels"], v.get("best_play_levels") or 0)):
        w, human = (i + 1) / sw, v["human_level_actions"][i]
        ours = v["level_actions"][i] if i < v["levels"] else None
        best = v["best_play_level_actions"][i] if i < (v.get("best_play_levels") or 0) else None
        pts = lambda a: min(115.0, 100 * (human / a) ** 2) * w if a else 0.0
        tot["human"] += 100 * w
        print(f"{g:5} {i + 1:5} {human:6} {ours or '-':>6} {pts(ours):7.2f} | {best or '-':>9} {pts(best):7.2f} | {100 * w:14.2f}")
k = len(games)
print(f"\nfirst exposure: {tot['first']:.2f} points -> score {tot['first'] / k:.2f}")
print(f"best play:      {tot['best']:.2f} points -> score {tot['best'] / k:.2f}   (per game capped by the toolkit; includes any clean run)")
print(f"solved levels at human speed would give {tot['human']:.2f} points -> {tot['human'] / k:.2f};  a score of 0.9 needs {0.9 * k:.1f} points")
