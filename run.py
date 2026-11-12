"""Evaluate an agent on ARC-AGI-3 games with the official scorecard metric.

uv run run.py [dev|heldout|all|<game> ...] [--agent v1|random] [--budget 1000] [--seeds 3] [--label NAME] [--off goal,deadline,replay5,llm | none]

--label NAME saves the full result to runs/<date>-NAME.json (the evidence file behind every reported number).

Metric (toolkit scorecard): per level min((human_baseline / actions)^2 * 100, 115), 0 if not completed;
per game a level-index-weighted mean capped at 100; overall = mean over games.
"""

import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":  # fixed hash seed -> identical runs (set order of byte-keyed types)
    os.execve(sys.executable, [sys.executable] + sys.argv, {**os.environ, "PYTHONHASHSEED": "0"})

import argparse
import datetime
import json
import logging
import statistics
import subprocess
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()
logging.disable(logging.CRITICAL)

import arc_agi
from arcengine import GameAction

import llm
from agent import Agent, Memory, RandomAgent

# Fixed split: tune on DEV only; HELDOUT is for final numbers (random.Random(0).sample of the 25 public games).
HELDOUT = ["bp35", "lf52", "lp85", "m0r0", "r11l", "re86", "sb26", "sc25", "su15", "wa30"]
DEV = ["ar25", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "ls20", "s5i5", "sk48", "sp80", "tn36", "tr87", "tu93", "vc33"]


def play(arc, game, agent, budget, seed=0):
    env = arc.make(game, seed=seed)
    f = env.reset()
    agent.new_game()
    g0, l0 = len(getattr(agent, "goal_log", [])), len(getattr(agent, "llm_log", []))
    actions = since = game_overs = 0
    per_level = []  # actions spent on each completed level (resets included)
    while actions < budget and f is not None:
        if f.state.name == "WIN":
            break
        if f.state.name == "GAME_OVER":
            game_overs += 1
            f = env.reset()
            agent.on_reset()
        else:
            act, data = agent.choose(f)
            f2 = env.step(GameAction[f"ACTION{act}"], data=data)
            if f2 is None:
                break
            agent.update(f, f2)
            if f2.levels_completed > f.levels_completed:
                per_level.append(since + 1)
                since = -1
            f = f2
        actions += 1
        since += 1
    return {"actions_per_level": per_level, "game_overs": game_overs, "goals": getattr(agent, "goal_log", [])[g0:],
            "llm": getattr(agent, "llm_log", [])[l0:]}


def run_seed(games, kind, budget, seed, off=None):
    """One full run: fresh scorecard, one analogy memory shared across all games."""
    arc = arc_agi.Arcade()
    mem = Memory()
    agent = Agent(mem, seed, off) if kind == "v1" else RandomAgent(seed)
    stats = [play(arc, g, agent, budget, seed) for g in games]
    return arc.get_scorecard(), agent, mem, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", default=["dev"])
    ap.add_argument("--agent", default="v1", choices=["v1", "random"])
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (game + agent), 0..n-1")
    ap.add_argument("--label", help="save the result to runs/<date>-<label>.json")
    ap.add_argument("--off", help="parts to switch off: goal, deadline, replay5, llm (comma-separated); 'none' = all on; "
                                  "default: the agent's own default (deadline and replay5 off, see protocol E12)")
    args = ap.parse_args()

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    # provenance is taken when the run starts, so later edits cannot change what the record claims
    prov = {"commit": git("rev-parse", "--short", "HEAD"),
            "uncommitted_code_changes": bool(git("status", "--porcelain", "--", "*.py"))}
    games = []
    for g in args.games:
        games += {"dev": DEV, "heldout": HELDOUT, "all": DEV + HELDOUT}.get(g, [g])

    off = None if args.off is None else [x for x in args.off.split(",") if x and x != "none"]
    runs = [run_seed(games, args.agent, args.budget, seed, off) for seed in range(args.seeds)]
    seeds = range(args.seeds)
    print(f"\n{'game':6} " + " ".join(f"{'s' + str(i):>9}" for i in seeds) + "   (levels / score per seed)")
    for j, g in enumerate(games):
        cells = []
        for sc, *_ in runs:
            e = sc.environments[j]
            cells.append(f"{e.levels_completed:2}/{e.score:5.2f}")
        print(f"{g:6} " + " ".join(f"{c:>9}" for c in cells))
    lv = [sum(e.levels_completed for e in sc.environments) for sc, *_ in runs]
    sco = [sc.score for sc, *_ in runs]
    apl = [n for *_, stats in runs for st in stats for n in st["actions_per_level"]]
    sd = lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0
    print(f"\nagent={args.agent} games={len(games)} budget={args.budget} seeds={args.seeds} off={sorted(Agent.DEFAULT_OFF) if off is None else off}")
    print(f"levels per seed {lv}  ->  mean {statistics.mean(lv):.2f} +/- {sd(lv):.2f}")
    print(f"score  per seed {[round(x, 2) for x in sco]}  ->  mean {statistics.mean(sco):.2f} +/- {sd(sco):.2f}")
    print(f"actions per completed level: median {statistics.median(apl) if apl else '-'}  (n={len(apl)})")
    print(f"game overs per seed {[sum(st['game_overs'] for st in stats) for *_, stats in runs]}")
    ranks = [g["rank"] for *_, stats in runs for st in stats for g in st["goals"] if g["rank"] is not None]
    learned = sum(g["schema"] is not None for *_, stats in runs for st in stats for g in st["goals"])
    print(f"goals learned from wins: {learned}; goal rank (0 best, 0.5 chance): "
          f"{'mean %.2f (n=%d)' % (statistics.mean(ranks), len(ranks)) if ranks else '- (no second level won)'}")
    asks = [a for *_, stats in runs for st in stats for a in st["llm"]]
    if asks:  # the LLM reasoner was used
        wins = [g for *_, stats in runs for st in stats for g in st["goals"] if g["llm_goal"]]
        print(f"LLM: {llm.CALLS['asked']} questions, {llm.CALLS['answered']} answered; goals proposed "
              f"{sum(a['proposed'] for a in asks)}, kept after the evidence check {sum(a['kept'] for a in asks)}; "
              f"levels won while testing an LLM goal {len(wins)}, of which the goal was true at the win "
              f"{sum(bool(g['llm_goal_true_at_win']) for g in wins)}")
    if args.agent == "v1":
        _, agent, mem, _ = runs[0]
        print("seed 0 skill usage:", dict(agent.usage.most_common()))
        print(f"seed 0 memory: {len(mem.roles)} object types, {len(mem.recipes)} recipes")
    if args.label:
        save(args, games, runs, lv, sco, apl, prov)


def save(args, games, runs, lv, sco, apl, prov):
    """Run record: everything needed to trace a reported number back to code + command."""
    sd = lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0
    record = {
        "label": args.label, "date": datetime.datetime.now().isoformat(timespec="seconds"),
        **prov,
        "command": "run.py " + " ".join(sys.argv[1:]), "agent": args.agent, "budget": args.budget, "games": games,
        "llm": {"model": os.environ.get("LLM_MODEL"), **llm.CALLS} if llm.enabled() else None,
        "summary": {"levels_per_seed": lv, "levels_mean": statistics.mean(lv), "levels_sd": sd(lv),
                    "score_per_seed": sco, "score_mean": statistics.mean(sco), "score_sd": sd(sco),
                    "actions_per_level_median": statistics.median(apl) if apl else None},
        "per_seed": [{
            "seed": i,
            "games": {g: {"levels": e.levels_completed, "level_count": e.level_count, "score": e.score,
                          "actions": e.actions, **st} for g, e, st in zip(games, sc.environments, stats)},
            "skill_usage": dict(agent.usage.most_common()),
            "memory": {"object_types": len(mem.roles), "recipes": len(mem.recipes)} if mem and args.agent == "v1" else None,
        } for i, (sc, agent, mem, stats) in enumerate(runs)],
    }
    Path("runs").mkdir(exist_ok=True)
    path = Path("runs") / f"{datetime.date.today().isoformat()}-{args.label}.json"
    if path.exists():  # never overwrite evidence
        path = path.with_name(f"{path.stem}-{datetime.datetime.now():%H%M%S}.json")
    path.write_text(json.dumps(record, indent=1))
    print(f"saved {path}")


if __name__ == "__main__":
    main()
