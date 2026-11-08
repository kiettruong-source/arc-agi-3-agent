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
    assert hud_bands(a, b)[15].all() and not hud_bands(g, b)[6, 6]


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


if __name__ == "__main__":
    test_perception()
    test_explore_then_exploit()
    print("ok")
