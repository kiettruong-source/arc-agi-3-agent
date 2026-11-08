"""Play an ARC-AGI-3 game by hand in the terminal.

Usage: uv run play.py [game_id]     (default ls20; `uv run play.py list` shows all)

Commands (then Enter):
  w/s/a/d or 1-4   ACTION1-4 (up/down/left/right)
  5 or e           ACTION5 (interact)
  6 X Y            ACTION6 click at grid x,y (0-63)
  7 or u           ACTION7 (undo)
  r                reset level      q   quit (prints scorecard)
"""

import sys

from dotenv import load_dotenv

load_dotenv()

import arc_agi
from arcengine import GameAction

KEYS = {"w": 1, "s": 2, "a": 3, "d": 4, "e": 5, "u": 7}

arc = arc_agi.Arcade()
game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
if game == "list":
    print("\n".join(e.game_id for e in arc.get_environments()))
    sys.exit()

env = arc.make(game, render_mode="terminal-fast")
f = env.reset()
while True:
    print(f"\n{f.game_id} | {f.state.name} | level {f.levels_completed}/{f.win_levels} | actions {f.available_actions}")
    parts = input("> ").strip().lower().split()
    if not parts:
        continue
    cmd = parts[0]
    if cmd == "q":
        break
    if cmd == "r":
        f = env.reset()
        continue
    n = KEYS.get(cmd) or (int(cmd) if cmd.isdigit() else None)
    if n not in f.available_actions:
        print(f"not available: {cmd}")
        continue
    data = None
    if n == 6:
        if len(parts) != 3:
            print("usage: 6 X Y")
            continue
        data = {"x": int(parts[1]), "y": int(parts[2])}
    f = env.step(GameAction[f"ACTION{n}"], data=data) or f

print(arc.get_scorecard())
