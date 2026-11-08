"""Evaluate an agent on ARC-AGI-3 games with the official scorecard metric.

uv run run.py [dev|heldout|all|<game> ...] [--agent v1|random] [--budget 1000] [--seed 0]

Metric (toolkit scorecard): per level min((human_baseline / actions)^2 * 100, 115), 0 if not completed;
per game a level-index-weighted mean capped at 100; overall = mean over games.
"""

import argparse
import logging

from dotenv import load_dotenv

load_dotenv()
logging.disable(logging.CRITICAL)

import arc_agi
from arcengine import GameAction

from agent import Agent, Memory, RandomAgent

# Fixed split: tune on DEV only; HELDOUT is for final numbers (random.Random(0).sample of the 25 public games).
HELDOUT = ["bp35", "lf52", "lp85", "m0r0", "r11l", "re86", "sb26", "sc25", "su15", "wa30"]
DEV = ["ar25", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "ls20", "s5i5", "sk48", "sp80", "tn36", "tr87", "tu93", "vc33"]


def play(arc, game, agent, budget):
    env = arc.make(game)
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", default=["dev"])
    ap.add_argument("--agent", default="v1", choices=["v1", "random"])
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    games = []
    for g in args.games:
        games += {"dev": DEV, "heldout": HELDOUT, "all": DEV + HELDOUT}.get(g, [g])

    arc = arc_agi.Arcade()
    mem = Memory()  # one analogy memory for the whole run
    agent = Agent(mem, args.seed) if args.agent == "v1" else RandomAgent(args.seed)
    for g in games:
        play(arc, g, agent, args.budget)

    sc = arc.get_scorecard()
    print(f"\n{'game':6} {'score':>6} {'levels':>7} {'actions':>7}")
    for e in sc.environments:
        print(f"{e.id[:4]:6} {e.score:6.2f} {e.levels_completed:3}/{e.level_count:<3} {e.actions:7}")
    total_levels = sum(e.levels_completed for e in sc.environments)
    print(f"\nagent={args.agent} games={len(sc.environments)} budget={args.budget}")
    print(f"OVERALL score={sc.score:.2f}  levels={total_levels}")
    if args.agent == "v1":
        print("skill usage:", dict(agent.usage.most_common()))
        print("skill value:", {k: round(v / n, 2) for k, (v, n) in mem.skill_stats.items()})
        print(f"memory: {len(mem.roles)} object types, {len(mem.recipes)} recipes")


if __name__ == "__main__":
    main()
