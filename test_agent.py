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


def collect_game(agent, levels, max_steps=120, cmap=None, flip=False):
    """Toy "collect" world: clicking a coin (colour 4) removes it; a level ends when no coin is left.
    Distractors (y, x, colour) never matter. `cmap` / `flip` show the same world with colours renamed and the
    grid mirrored. Returns the number of actions used per level."""
    def scene(coins, junk):
        g = np.zeros((16, 16), int)
        for y, x in coins:
            g[y, x] = 4
        for y, x, c in junk:
            g[y:y + 2, x:x + 2] = c
        if cmap:
            g = np.vectorize(lambda v: cmap.get(v, v))(g)
        return np.fliplr(g) if flip else g
    used, lvl = [0] * len(levels), 0
    coins, junk = list(levels[0][0]), levels[0][1]
    f = frame(scene(coins, junk))
    for _ in range(max_steps):
        act, data = agent.choose(f)
        used[lvl] += 1
        pos = (data["y"], 15 - data["x"] if flip else data["x"])
        if pos in coins:
            coins.remove(pos)
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


# level 2 has more coins and six eye-catching distractors (each a rare colour, which plain exploring tries first)
COLLECT = [([(2, 2), (2, 8), (9, 4)], [(12, 10, 7)]),
           ([(1, 1), (1, 6), (5, 12), (9, 3), (13, 8)],
            [(11, 12, 7), (6, 6, 2), (3, 10, 3), (8, 8, 5), (13, 2, 6), (5, 3, 8), (10, 13, 9)])]


def test_goal_from_win():
    agent = Agent()
    used = collect_game(agent, COLLECT)
    assert len(agent.goal_log) == 2, agent.goal_log
    schema = agent.mem.goals[0][0]
    d, to_zero, mag, weight = schema[("count", "colour", 4)]
    assert (d, to_zero, weight) == (-1, True, 2), schema          # "no coin left" was never seen before the win
    assert ("count", "colour", 7) not in schema, schema            # distractors are not part of the goal
    assert agent.goal_log[1]["rank"] == 0.0, agent.goal_log        # level 1's goal picks out level 2's win screen
    # using the goal: level 2 (5 coins among 7 distractors) takes about one click per coin
    blind = collect_game(Agent(off={"goal"}), COLLECT)
    assert used[1] <= 7 and used[1] <= blind[1], (used, blind)


def test_general_reasoning():
    # same reasoning code, a different kind of world: walk to a target (colour 3); distractors lie nearer.
    def reach_game(agent, max_steps=200):
        levels = [dict(player=(1, 5), target=(10, 5), walls=[(6, x) for x in range(1, 11)], junk=[]),
                  dict(player=(1, 1), target=(10, 10), walls=[], junk=[(1, 4, 6), (4, 1, 7), (3, 3, 8)])]
        def scene(L, pos):
            g = np.zeros((12, 12), int)
            for y, x in L["walls"]:
                g[y, x] = 5
            for y, x, c in L["junk"]:
                g[y, x] = c
            g[L["target"]] = 3
            g[pos] = 2
            return g
        used, lvl = [0, 0], 0
        L, pos = levels[0], levels[0]["player"]
        f = frame(scene(L, pos), actions=(1, 2, 3, 4))
        for _ in range(max_steps):
            act, _ = agent.choose(f)
            used[lvl] += 1
            dy, dx = {1: (-1, 0), 2: (1, 0), 3: (0, -1), 4: (0, 1)}[act]
            ny, nx = pos[0] + dy, pos[1] + dx
            if 0 <= ny < 12 and 0 <= nx < 12 and scene(L, pos)[ny, nx] == 0:
                pos = (ny, nx)
            if abs(pos[0] - L["target"][0]) + abs(pos[1] - L["target"][1]) > 1:
                f2 = frame(scene(L, pos), levels=lvl, actions=(1, 2, 3, 4))
            else:
                won, lvl = scene(L, pos), lvl + 1
                if lvl == 2:
                    agent.update(f, frame(won, levels=2, state="WIN", actions=(1, 2, 3, 4)))
                    break
                L, pos = levels[1], levels[1]["player"]
                f2 = frame(scene(L, pos), levels=1, actions=(1, 2, 3, 4), before=won)
            agent.update(f, f2)
            f = f2
        return used
    agent = Agent()
    used, blind = reach_game(agent), reach_game(Agent(off={"goal"}))
    schema = agent.mem.goals[0][0]
    assert schema[("dist", "colour", 3)][:2] == (-1, True), schema     # "touching the target" learned by contrast
    assert used[1] < blind[1] and used[1] <= 24, (used, blind)         # shortest route is 17 moves
    # same world with colours renamed and the grid mirrored: the same number of actions
    plain = collect_game(Agent(), COLLECT)
    renamed = collect_game(Agent(), COLLECT, cmap={4: 9, 9: 4, 7: 1, 2: 12, 3: 11}, flip=True)
    assert plain == renamed, (plain, renamed)


def test_timeout_is_not_deadly():
    # a click world with a step limit of 10 and no way to win: no object may end up "deadly"
    def run(agent):
        g = np.zeros((16, 16), int)
        for i in range(3):
            g[12:14, 2 + 5 * i:4 + 5 * i] = 6
            g[3, 3 + 5 * i] = 9
        f = frame(g)
        for step in range(1, 36):
            act, data = agent.choose(f)
            g2 = np.array(f.frame[-1])
            if g2[data["y"], data["x"]] == 6:
                i = (data["x"] - 2) // 5
                g2[3, 3 + 5 * i] = 2 if g2[3, 3 + 5 * i] == 9 else 9
            dead = step % 10 == 0
            agent.update(f, frame(g2, state="GAME_OVER" if dead else "NOT_FINISHED"))
            if dead:
                agent.on_reset()
                g2 = g
            f = frame(g2)
        return sum(r["deaths"] for r in agent.mem.roles.values())
    assert run(Agent()) == 0
    assert run(Agent(off={"deadline"})) > 0   # the old rule blamed whatever was clicked last


if __name__ == "__main__":
    test_goal_from_win()
    test_general_reasoning()
    test_timeout_is_not_deadly()
    test_frontier_exploration()
    test_structure_mapping()
    test_perception()
    test_explore_then_exploit()
    test_map_planning()
    print("ok")
