"""Smallest checks that break if perception or the explore -> exploit switch breaks. Run: uv run python test_agent.py"""

from types import SimpleNamespace as NS

import numpy as np

from agent import Agent, hud_bands, objects


def frame(grid, levels=0, state="NOT_FINISHED", actions=(6,)):
    return NS(frame=[grid.tolist()], levels_completed=levels, state=NS(name=state), available_actions=list(actions))


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


if __name__ == "__main__":
    test_structure_mapping()
    test_perception()
    test_explore_then_exploit()
    test_map_planning()
    print("ok")
