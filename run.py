"""Evaluate an agent on ARC-AGI-3 games with the official scorecard metric.

uv run run.py [dev|heldout|all|<game> ...] [--agent v1|random] [--budget 1000] [--seeds 3]

Metric (toolkit scorecard): per level min((human_baseline / actions)^2 * 100, 115), 0 if not completed;
per game a level-index-weighted mean capped at 100; overall = mean over games.
"""

import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":  # fixed hash seed -> identical runs (set order of byte-keyed types)
    os.execve(sys.executable, [sys.executable] + sys.argv, {**os.environ, "PYTHONHASHSEED": "0"})

import argparse
import logging
import statistics

from dotenv import load_dotenv

load_dotenv()
logging.disable(logging.CRITICAL)

import arc_agi
from arcengine import GameAction

from agent import Agent, Memory, RandomAgent

# Fixed split: tune on DEV only; HELDOUT is for final numbers (random.Random(0).sample of the 25 public games).
HELDOUT = ["bp35", "lf52", "lp85", "m0r0", "r11l", "re86", "sb26", "sc25", "su15", "wa30"]
DEV = ["ar25", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "ls20", "s5i5", "sk48", "sp80", "tn36", "tr87", "tu93", "vc33"]


def play(arc, game, agent, budget, seed=0):
    env = arc.make(game, seed=seed)
    f = env.reset()
    agent.new_game()
    actions = 0
    while actions < budget and f is not None:
        if f.state.name == "WIN":
            break
        if f.state.name == "GAME_OVER":
            f = env.reset()
            agent.on_reset()
        else:
            act, data = agent.choose(f)
            f2 = env.step(GameAction[f"ACTION{act}"], data=data)
            if f2 is None:
                break
            agent.update(f, f2)
            f = f2
        actions += 1


def run_seed(games, kind, budget, seed):
    """One full run: fresh scorecard, one analogy memory shared across all games."""
    arc = arc_agi.Arcade()
    mem = Memory()
    agent = Agent(mem, seed) if kind == "v1" else RandomAgent(seed)
    for g in games:
        play(arc, g, agent, budget, seed)
    return arc.get_scorecard(), agent, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", default=["dev"])
    ap.add_argument("--agent", default="v1", choices=["v1", "random"])
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (game + agent), 0..n-1")
    args = ap.parse_args()

    games = []
    for g in args.games:
        games += {"dev": DEV, "heldout": HELDOUT, "all": DEV + HELDOUT}.get(g, [g])

    runs = [run_seed(games, args.agent, args.budget, seed) for seed in range(args.seeds)]
    seeds = range(args.seeds)
    print(f"\n{'game':6} " + " ".join(f"{'s' + str(i):>9}" for i in seeds) + "   (levels / score per seed)")
    for j, g in enumerate(games):
        cells = []
        for sc, _, _ in runs:
            e = sc.environments[j]
            cells.append(f"{e.levels_completed:2}/{e.score:5.2f}")
        print(f"{g:6} " + " ".join(f"{c:>9}" for c in cells))
    lv = [sum(e.levels_completed for e in sc.environments) for sc, _, _ in runs]
    sco = [sc.score for sc, _, _ in runs]
    sd = lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0
    print(f"\nagent={args.agent} games={len(games)} budget={args.budget} seeds={args.seeds}")
    print(f"levels per seed {lv}  ->  mean {statistics.mean(lv):.2f} +/- {sd(lv):.2f}")
    print(f"score  per seed {[round(x, 2) for x in sco]}  ->  mean {statistics.mean(sco):.2f} +/- {sd(sco):.2f}")
    if args.agent == "v1":
        _, agent, mem = runs[0]
        print("seed 0 skill usage:", dict(agent.usage.most_common()))
        print(f"seed 0 memory: {len(mem.roles)} object types, {len(mem.recipes)} recipes")


if __name__ == "__main__":
    main()
