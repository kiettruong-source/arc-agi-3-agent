"""Evaluate an agent on ARC-AGI-3 games with the official scorecard metric.

uv run run.py [dev|heldout|all|<game> ...] [--agent v1|random] [--budget 1000] [--seeds 3] [--label NAME] [--off goal,deadline,replay5,reset,cluster,llm | none] [--clean-run] [--rename N] [--fresh-memory]

--label NAME saves the full result to runs/<date>-NAME.json (the evidence file behind every reported number).
--clean-run  after the normal play (the "first exposure" numbers), a second play replays the shortest known
             solution of every level the agent solved. The toolkit scores the best play of a game, so this gives a
             second number, "best play": it says how short the found solutions are, not how efficiently the agent
             learns. Both numbers are always reported separately.
--rename N   robustness test: every game is shown with its colours renamed (permutation number N) and the grid
             mirrored. Nothing about the game changes; only what it looks like.
--fresh-memory   transfer test: the agent starts every game with an empty memory (default: memory carried over).

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
import random
import statistics
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv

load_dotenv()
logging.disable(logging.CRITICAL)

import arc_agi
import numpy as np
from arcengine import GameAction

import llm
from agent import Agent, Memory, RandomAgent

# Fixed split: tune on DEV only; HELDOUT is for final numbers (random.Random(0).sample of the 25 public games).
HELDOUT = ["bp35", "lf52", "lp85", "m0r0", "r11l", "re86", "sb26", "sc25", "su15", "wa30"]
DEV = ["ar25", "cd82", "cn04", "dc22", "ft09", "g50t", "ka59", "ls20", "s5i5", "sk48", "sp80", "tn36", "tr87", "tu93", "vc33"]


class Renamed:
    """The same game shown differently: colours renamed by a fixed permutation, grid mirrored left-right.
    Clicks are mapped back. An agent that reasons about structure should play it just as well."""

    def __init__(self, env, n):
        self.env, self.perm, self.width = env, np.array(random.Random(n).sample(range(16), 16)), None

    def view(self, f):
        if f is None:
            return None
        frames = [np.fliplr(self.perm[np.asarray(g)]) for g in f.frame]
        if frames:
            self.width = frames[-1].shape[1]
        return SimpleNamespace(frame=frames, state=f.state, levels_completed=f.levels_completed,
                               available_actions=f.available_actions)

    def reset(self):
        return self.view(self.env.reset())

    def step(self, action, data=None):
        if data and self.width:
            data = {**data, "x": self.width - 1 - data["x"]}
        return self.view(self.env.step(action, data=data))


def play(make, agent, budget):
    """One play of one game. `make()` returns an environment (reset / step)."""
    env = make()
    f = env.reset()
    agent.new_game()
    g0, l0 = len(getattr(agent, "goal_log", [])), len(getattr(agent, "llm_log", []))
    actions = since = game_overs = 0
    thinking = 0.0  # seconds the agent itself needs (choose + learn), the game's own time excluded
    per_level = []  # actions spent on each completed level (resets included)
    while actions < budget and f is not None:
        if f.state.name == "WIN":
            break
        if f.state.name == "GAME_OVER":
            game_overs += 1
            f = env.reset()
            agent.on_reset()
        else:
            t0 = time.perf_counter()
            act, data = agent.choose(f)
            thinking += time.perf_counter() - t0
            if act == 0:  # the agent asks for a level reset (a 1-action jump back to the level's start)
                f = env.reset()
                agent.on_reset()
                actions += 1
                since += 1
                continue
            f2 = env.step(GameAction[f"ACTION{act}"], data=data)
            if f2 is None:
                break
            t0 = time.perf_counter()
            agent.update(f, f2)
            thinking += time.perf_counter() - t0
            if f2.levels_completed > f.levels_completed:
                per_level.append(since + 1)
                since = -1
            f = f2
        actions += 1
        since += 1
    return {"actions_per_level": per_level, "game_overs": game_overs, "goals": getattr(agent, "goal_log", [])[g0:],
            "llm": getattr(agent, "llm_log", [])[l0:], "agent_seconds": round(thinking, 2), "played_actions": actions}


def clean_run(make, agent):
    """A fresh play that replays what the agent found: per level the shortest known route, checked after every
    step; on a mismatch one level reset and the attempt that actually won. Stops at the first level it cannot redo."""
    env = make()
    f = env.reset()
    used = done = 0
    for sol in agent.solutions:
        level, tried = f.levels_completed, False
        for steps in (sol["short"], sol["actual"]):
            if not steps:
                continue
            if tried:  # the short route failed: back to the level's start
                f = env.reset()
                used += 1
            tried = True
            for (act, data), expect in steps:
                f2 = env.step(GameAction[f"ACTION{act}"], data=data)
                used += 1
                if f2 is None:
                    break
                f = f2
                if f.levels_completed > level or f.state.name in ("GAME_OVER", "WIN"):
                    break
                if expect is not None and not agent.same_screen(np.array(f.frame[-1]), expect):
                    break
            if f.levels_completed > level:
                break
        if f.levels_completed <= level:
            break
        done += 1
        if f.state.name == "WIN":
            break
    return {"levels": done, "actions": used}


def first_play(e):
    """The first real play of a game on the scorecard (the toolkit also lists empty plays)."""
    return next((r for r in e.runs if r.actions), e.runs[0])


def run_seed(games, kind, budget, seed, off=None, clean=False, rename=0, fresh=False):
    """One full run: fresh scorecard, one analogy memory shared across all games (unless `fresh`)."""
    arc = arc_agi.Arcade()
    mem = Memory()
    agent = Agent(mem, seed, off) if kind == "v1" else RandomAgent(seed)
    stats = []
    for g in games:
        if fresh and kind == "v1":
            agent.mem = mem = Memory()  # transfer test: nothing carried over from earlier games
        make = (lambda g=g: Renamed(arc.make(g, seed=seed), rename)) if rename else (lambda g=g: arc.make(g, seed=seed))
        st = play(make, agent, budget)
        st["clean"] = clean_run(make, agent) if clean and getattr(agent, "solutions", None) else None
        stats.append(st)
    return arc.get_scorecard(), agent, mem, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("games", nargs="*", default=["dev"])
    ap.add_argument("--agent", default="v1", choices=["v1", "random"])
    ap.add_argument("--budget", type=int, default=1000)
    ap.add_argument("--seeds", type=int, default=1, help="number of seeds (game + agent), 0..n-1")
    ap.add_argument("--label", help="save the result to runs/<date>-<label>.json")
    ap.add_argument("--clean-run", action="store_true", help="add a second play that replays the found solutions")
    ap.add_argument("--rename", type=int, default=0, help="show every game with colours renamed (permutation N) and mirrored")
    ap.add_argument("--fresh-memory", action="store_true", help="empty memory at the start of every game")
    ap.add_argument("--off", help="parts to switch off: goal, deadline, replay5, reset, cluster, llm (comma-separated); 'none' = all on; "
                                  "default: the agent's own default (see Agent.DEFAULT_OFF and the protocol)")
    args = ap.parse_args()

    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    # provenance is taken when the run starts, so later edits cannot change what the record claims
    prov = {"commit": git("rev-parse", "--short", "HEAD"),
            "uncommitted_code_changes": bool(git("status", "--porcelain", "--", "*.py"))}
    games = []
    for g in args.games:
        games += {"dev": DEV, "heldout": HELDOUT, "all": DEV + HELDOUT}.get(g, [g])

    off = None if args.off is None else [x for x in args.off.split(",") if x and x != "none"]
    runs = [run_seed(games, args.agent, args.budget, seed, off, args.clean_run, args.rename, args.fresh_memory)
            for seed in range(args.seeds)]
    seeds = range(args.seeds)
    print(f"\n{'game':6} " + " ".join(f"{'s' + str(i):>9}" for i in seeds) + "   (first exposure: levels / score per seed)")
    for j, g in enumerate(games):
        cells = []
        for sc, *_ in runs:
            r = first_play(sc.environments[j])
            cells.append(f"{r.levels_completed:2}/{r.score:5.2f}")
        print(f"{g:6} " + " ".join(f"{c:>9}" for c in cells))
    lv = [sum(first_play(e).levels_completed for e in sc.environments) for sc, *_ in runs]
    sco = [statistics.mean(first_play(e).score for e in sc.environments) for sc, *_ in runs]
    best = [sc.score for sc, *_ in runs]
    apl = [n for *_, stats in runs for st in stats for n in st["actions_per_level"]]
    sd = lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0
    print(f"\nagent={args.agent} games={len(games)} budget={args.budget} seeds={args.seeds} off={sorted(Agent.DEFAULT_OFF) if off is None else off}")
    print("FIRST EXPOSURE (every action from the first frame counts):")
    print(f"levels per seed {lv}  ->  mean {statistics.mean(lv):.2f} +/- {sd(lv):.2f}")
    print(f"score  per seed {[round(x, 2) for x in sco]}  ->  mean {statistics.mean(sco):.2f} +/- {sd(sco):.2f}")
    if args.clean_run:
        cl = [[st["clean"] for st in stats if st["clean"]] for *_, stats in runs]
        print("BEST PLAY (with a clean run: replay of the found solutions; not a measure of learning efficiency):")
        print(f"score  per seed {[round(x, 2) for x in best]}  ->  mean {statistics.mean(best):.2f} +/- {sd(best):.2f}")
        print(f"clean-run levels redone per seed {[sum(c['levels'] for c in s) for s in cl]}, "
              f"extra actions per seed {[sum(c['actions'] for c in s) for s in cl]}")
    print(f"actions per completed level: median {statistics.median(apl) if apl else '-'}  (n={len(apl)})")
    print(f"game overs per seed {[sum(st['game_overs'] for st in stats) for *_, stats in runs]}")
    ranks = [g["rank"] for *_, stats in runs for st in stats for g in st["goals"] if g["rank"] is not None]
    learned = sum(g["schema"] is not None for *_, stats in runs for st in stats for g in st["goals"])
    secs = sum(st["agent_seconds"] for *_, stats in runs for st in stats)
    acts = sum(st["played_actions"] for *_, stats in runs for st in stats)
    print(f"cost: {1000 * secs / max(1, acts):.1f} ms of agent time per action ({secs:.0f} s for {acts} actions); "
          f"LLM calls {llm.CALLS['asked']}")
    tested = [g for *_, stats in runs for st in stats for g in st["goals"] if g.get("goal_kind")]
    print(f"goal precision: {sum(bool(g['goal_true_at_win']) for g in tested)} of {len(tested)} levels won with a goal "
          f"to test had that goal true on the winning screen (given up before the win: {sum(g['goal_given_up'] for g in tested)})")
    if args.agent == "v1":
        pr = sum((a.pred for _, a, *_ in runs), start=type(runs[0][1].pred)())
        right = sum(n for k, n in pr.items() if k.endswith(" right"))
        wrong = sum(n for k, n in pr.items() if k.endswith(" wrong"))
        base = pr["changed"] / max(1, pr["actions"])
        rep = pr["repeat same"] + pr["repeat different"]
        print(f"prediction (will the action change the screen?): made for {(right + wrong) / max(1, pr['actions']):.0%} of actions, "
              f"right {right / max(1, right + wrong):.0%} (always guessing the commoner outcome: {max(base, 1 - base):.0%}); "
              f"same action on the same screen gave the same screen {pr['repeat same'] / max(1, rep):.0%} of {rep} times")
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
        save(args, games, runs, lv, sco, apl, prov, best)


def save(args, games, runs, lv, sco, apl, prov, best):
    """Run record: everything needed to trace a reported number back to code + command."""
    sd = lambda xs: statistics.stdev(xs) if len(xs) > 1 else 0.0
    record = {
        "label": args.label, "date": datetime.datetime.now().isoformat(timespec="seconds"),
        **prov,
        "command": "run.py " + " ".join(sys.argv[1:]), "agent": args.agent, "budget": args.budget, "games": games,
        "rename": args.rename, "fresh_memory": args.fresh_memory,
        "llm": {"model": os.environ.get("LLM_MODEL"), **llm.CALLS} if llm.enabled() else None,
        "summary": {"levels_per_seed": lv, "levels_mean": statistics.mean(lv), "levels_sd": sd(lv),
                    "score_per_seed": sco, "score_mean": statistics.mean(sco), "score_sd": sd(sco),
                    "actions_per_level_median": statistics.median(apl) if apl else None,
                    "note": "levels / score = first exposure (first play of each game)",
                    "best_play_score_per_seed": best if args.clean_run else None,
                    "best_play_score_mean": statistics.mean(best) if args.clean_run else None},
        "per_seed": [{
            "seed": i,
            "games": {g: {"levels": first_play(e).levels_completed, "level_count": e.level_count,
                          "score": first_play(e).score, "actions": first_play(e).actions,
                          "level_actions": first_play(e).level_actions, "level_scores": first_play(e).level_scores,
                          "human_level_actions": first_play(e).level_baseline_actions,
                          "best_play_score": e.score,
                          "best_play_levels": max(e.runs, key=lambda r: r.score).levels_completed,
                          "best_play_level_actions": max(e.runs, key=lambda r: r.score).level_actions,
                          **st} for g, e, st in zip(games, sc.environments, stats)},
            "skill_usage": dict(agent.usage.most_common()),
            "prediction": dict(getattr(agent, "pred", {})),
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
