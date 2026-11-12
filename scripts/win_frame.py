"""E10: which of the frames returned by one action shows the level that was just won (or lost)?

uv run python scripts/win_frame.py [game ...]      (default: the dev games v4 completes a level in)

For every level-up and game over it prints how many frames the action returned and how much each frame differs
from the screen before the action (share of cells). A frame close to the previous screen still shows the old
level; a large jump marks the switch to the next level.
"""
import os
import sys

if os.environ.get("PYTHONHASHSEED") != "0":
    os.execve(sys.executable, [sys.executable] + sys.argv, {**os.environ, "PYTHONHASHSEED": "0"})

import logging
from collections import Counter

logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
import arc_agi
import numpy as np
from arcengine import GameAction

from agent import Agent

games = sys.argv[1:] or ["cd82", "ls20", "sp80", "tn36", "vc33"]
arc = arc_agi.Arcade()
for game in games:
    agent = Agent()
    env = arc.make(game)
    f = env.reset()
    agent.new_game()
    n_frames = Counter()
    for step in range(600):
        if f.state.name == "WIN":
            break
        if f.state.name == "GAME_OVER":
            f = env.reset()
            agent.on_reset()
            continue
        act, data = agent.choose(f)
        if act == 0:  # deliberate level reset
            f = env.reset()
            agent.on_reset()
            continue
        f2 = env.step(GameAction[f"ACTION{act}"], data=data)
        if f2 is None:
            break
        n_frames[len(f2.frame)] += 1
        up, dead = f2.levels_completed > f.levels_completed, f2.state.name == "GAME_OVER"
        if up or dead:
            prev = np.array(f.frame[-1])
            diffs = [round(float((np.array(x) != prev).mean()), 3) if np.array(x).shape == prev.shape else None
                     for x in f2.frame]
            print(f"{game} step {step:3} {'LEVEL-UP' if up else 'GAME-OVER':9} frames={len(f2.frame)} diff to previous screen per frame: {diffs}")
        agent.update(f, f2)
        f = f2
    print(f"{game} frames per action (count): {dict(sorted(n_frames.items()))}\n")
