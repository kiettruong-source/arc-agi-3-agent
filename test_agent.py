"""Smallest checks that break if perception or the explore -> exploit switch breaks. Run: uv run python test_agent.py"""

from types import SimpleNamespace as NS

import numpy as np

from agent import Agent, hud_bands, objects


def frame(grid, levels=0, state="NOT_FINISHED", actions=(6,), before=None):
    """`before` = the winning screen of the level just completed (the engine sends it ahead of the new level)."""
    frames = ([before.tolist()] if before is not None else []) + [grid.tolist()]
    return NS(frame=frames, levels_completed=levels, state=NS(name=state), available_actions=list(actions))


def test_perception():
    g = np.zeros((16, 16), int)
    g[2:4, 2:4] = 3   # two identical 2x2 squares -> one type
    g[10:12, 8:10] = 3
    g[6, 6] = 7       # a single odd pixel -> another type
    objs = objects(g, np.zeros(g.shape, bool))
    assert len(objs) == 3 and len({o.type for o in objs}) == 2
    # a bar shrinking at the bottom edge is HUD; a change in the middle is not
    a, b = g.copy(), g.copy()
    a[15, 0:10] = 5
    b[15, 0:9] = 5
    assert hud_bands(a, b)[15].all() and not hud_bands(g, b)[6, 6] and not hud_bands(a, b, [(15, 0, 15, 9)]).any()


def test_explore_then_exploit():
    # toy click game: three yellow pixels that toggle to blue when clicked
    g = np.zeros((16, 16), int)
    cells = [(3, 3), (3, 9), (9, 6)]
    for y, x in cells:
        g[y, x] = 4
    agent = Agent()
    f = frame(g)
    used = []
    for _ in range(4):
        act, data = agent.choose(f)
        used.append(agent.pending[2])
        g2 = np.array(f.frame[-1])
        if act == 6 and g2[data["y"], data["x"]] == 4:
            g2[data["y"], data["x"]] = 1
        f2 = frame(g2)
        agent.update(f, f2)
        f = f2
    assert used[0] == "probe", used                 # unknown type -> explore first
    assert "click_all" in used, used                 # learned "yellow does something" -> exploit on the rest
    assert (np.array(f.frame[-1]) == 4).sum() == 0   # all three toggled


def test_map_planning():
    # toy maze: target straight below the player behind a wall -> greedy gets stuck, planning walks around
    g = np.zeros((12, 12), int)
    g[6, 1:11] = 5          # wall with gaps at both ends
    g[1, 5] = 2             # player
    g[10, 5] = 3            # target
    agent = Agent()
    f = frame(g, actions=(1, 2, 3, 4))
    for step in range(40):
        act, _ = agent.choose(f)
        g2 = np.array(f.frame[-1])
        y, x = map(int, np.argwhere(g2 == 2)[0])
        dy, dx = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[act]
        ny, nx = y + dy, x + dx
        if 0 <= ny < 12 and 0 <= nx < 12 and g2[ny, nx] in (0, 3):
            g2[y, x], g2[ny, nx] = 0, 2
        f2 = frame(g2, actions=(1, 2, 3, 4))
        agent.update(f, f2)
        f = f2
        if abs(ny - 10) + abs(nx - 5) <= 1 and g2[ny, nx] == 2:
            break
    assert step < 30, step                          # reached the target well within budget
    assert "go_to" in agent.usage, agent.usage      # by planning, not by luck


def test_structure_mapping():
    # level 1: three grey squares + one red pixel; clicking the odd one out wins.
    # level 2: colours AND positions change (blue squares, green pixel): replay must map by role, not looks.
    def level(sq, odd, pos):
        g = np.zeros((16, 16), int)
        for y, x in ((2, 2), (2, 10), (10, 2)):
            g[y:y + 2, x:x + 2] = sq
        g[pos] = odd
        return g
    agent = Agent()
    f = frame(level(8, 2, (12, 12)))
    act, data = agent.choose(f)
    assert (data["y"], data["x"]) == (12, 12), data               # rare + small first -> the odd pixel
    f2 = frame(level(1, 3, (5, 7)), levels=1)                      # it was the winning click
    agent.update(f, f2)
    act, data = agent.choose(f2)
    assert agent.pending[2] == "replay", agent.pending[2]
    assert (data["y"], data["x"]) == (5, 7), data                 # mapped to the new odd one out


def test_frontier_exploration():
    # lights puzzle: 3 identical buttons each toggle their own lamp; the level is won only at lamps (on, off, on).
    # probe tries the button *type* once, click_all presses all three -> (on, on, on), not a win.
    # graph exploration must then search combinations instead of clicking randomly.
    def scene(lamps):
        g = np.zeros((16, 16), int)
        for i, on in enumerate(lamps):
            g[12:14, 2 + 5 * i:4 + 5 * i] = 6          # button i (same type for all three)
            g[3, 3 + 5 * i] = 2 if on else 9           # lamp i
        return g
    agent, lamps, won = Agent(), [0, 0, 0], False
    f = frame(scene(lamps))
    for step in range(30):
        act, data = agent.choose(f)
        i = (data["x"] - 2) // 5 if 12 <= data["y"] <= 13 else None
        if i is not None and 0 <= i < 3 and data["x"] <= 3 + 5 * i:
            lamps[i] ^= 1
        won = lamps == [1, 0, 1]
        f2 = frame(scene(lamps), levels=int(won))
        agent.update(f, f2)
        f = f2
        if won:
            break
    assert won, (lamps, agent.usage)
    assert "frontier" in agent.usage and "random" not in agent.usage, agent.usage


def collect_game(agent, levels, max_steps=80):
    """Toy "collect" world: clicking a coin (colour 4) removes it; a level ends when no coin is left.
    Distractors (colour 7) never matter. Returns the number of actions used per level."""
    def scene(coins, junk):
        g = np.zeros((16, 16), int)
        for y, x in coins:
            g[y, x] = 4
        for y, x in junk:
            g[y:y + 2, x:x + 2] = 7
        return g
    used, lvl = [0] * len(levels), 0
    coins, junk = list(levels[0][0]), levels[0][1]
    f = frame(scene(coins, junk))
    for _ in range(max_steps):
        act, data = agent.choose(f)
        used[lvl] += 1
        if (data["y"], data["x"]) in coins:
            coins.remove((data["y"], data["x"]))
        if coins:
            f2 = frame(scene(coins, junk), levels=lvl)
        else:
            won, lvl = scene(coins, junk), lvl + 1
            if lvl == len(levels):
                agent.update(f, frame(won, levels=lvl, state="WIN"))
                break
            coins, junk = list(levels[lvl][0]), levels[lvl][1]
            f2 = frame(scene(coins, junk), levels=lvl, before=won)
        agent.update(f, f2)
        f = f2
    return used


COLLECT = [([(2, 2), (2, 8), (9, 4)], [(12, 10)]),
           ([(1, 1), (1, 6), (5, 12), (9, 3), (13, 8)], [(11, 12), (6, 6)])]


def test_goal_from_win():
    agent = Agent()
    collect_game(agent, COLLECT)
    assert len(agent.goal_log) == 2, agent.goal_log
    schema = agent.mem.goals[0][0]
    d, to_zero, mag, weight = schema[("count", "colour", 4)]
    assert (d, to_zero, weight) == (-1, True, 2), schema          # "no coin left" was never seen before the win
    assert ("count", "colour", 7) not in schema, schema            # distractors are not part of the goal
    assert agent.goal_log[1]["rank"] == 0.0, agent.goal_log        # level 1's goal picks out level 2's win screen


if __name__ == "__main__":
    test_goal_from_win()
    test_frontier_exploration()
    test_structure_mapping()
    test_perception()
    test_explore_then_exploit()
    test_map_planning()
    print("ok")
