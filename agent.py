"""Minimal ARC-AGI-3 agent v1: perception -> fundamental skills -> explore / exploit, with an analogy memory.

The agent only uses what every game gives: a colour grid, the available actions, and level / game-over
signals. No game ids, no fixed grid size, no meaning attached to any colour.
"""

import random
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import label

MOVES = (1, 2, 3, 4)
CLICK = 6


# ---------------------------------------------------------------- perception
@dataclass(frozen=True)
class Obj:
    type: tuple  # (colour, h, w, shape bytes): same thing anywhere on the grid
    colour: int
    x: int  # a cell that is really on the object (click target)
    y: int
    cx: float  # centroid
    cy: float
    size: int
    box: tuple  # y0, x0, y1, x1


def objects(grid, hud):
    """Same-colour connected components, background and HUD excluded."""
    bg = Counter(grid.ravel().tolist()).most_common(1)[0][0]
    out = []
    for c in np.unique(grid):
        if c == bg:
            continue
        lab, n = label((grid == c) & ~hud)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            m = lab[ys.min():ys.max() + 1, xs.min():xs.max() + 1] == i
            k = len(ys) // 2
            out.append(Obj((int(c), *m.shape, m.tobytes()), int(c), int(xs[k]), int(ys[k]),
                           float(xs.mean()), float(ys.mean()), len(ys),
                           (int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max()))))
    return out


def hud_bands(prev, new, exclude=None):
    """Thin changes hugging the border = counters / bars. Returns a mask of whole bands (a HUD vote).
    `exclude` = box (y0, x0, y1, x1) of the object just clicked: its own change is an effect, not HUD."""
    mask = np.zeros(new.shape, bool)
    lab, n = label(prev != new, structure=np.ones((3, 3)))
    H, W = new.shape
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        if exclude and not (y1 < exclude[0] or y0 > exclude[2] or x1 < exclude[1] or x0 > exclude[3]):
            continue
        # ponytail: border+thin heuristic; a player walking along the edge could be masked. Learn per-cell stats if it bites.
        if y1 - y0 <= 2 and (y0 <= 3 or y1 >= H - 4):
            mask[y0:y1 + 1, :] = True
        elif x1 - x0 <= 2 and (x0 <= 3 or x1 >= W - 4):
            mask[:, x0:x1 + 1] = True
    return mask


# ---------------------------------------------------------------- memory
@dataclass
class Memory:
    """Analogy memory, shared across levels and games in one run."""
    roles: dict = field(default_factory=dict)    # type -> {"tries", "effects", "deaths"}
    recipes: list = field(default_factory=list)  # [[(action, target type | None), ...], ...]
    skill_stats: dict = field(default_factory=dict)  # name -> [value_sum, steps]

    def role(self, t):
        return self.roles.setdefault(t, {"tries": 0, "effects": 0, "deaths": 0})

    def known(self, t):
        return t in self.roles and self.roles[t]["tries"] > 0

    def analog(self, t, objs):
        """Objects matching type t: exact type first, else same colour (the analogy fallback)."""
        exact = [o for o in objs if o.type == t]
        return exact or [o for o in objs if o.colour == t[0]]


# ---------------------------------------------------------------- skills
# Each skill: applicable(agent) -> bool, next(agent) -> (action, obj | None) or None when finished.
class Skill:
    name = kind = ""

    def applicable(self, a):
        return True

    def start(self, a):
        pass


class Probe(Skill):
    """Explore: act once per untested thing. Unknown types first, salient (rare colour, small) first."""
    name, kind = "probe", "explore"

    def candidates(self, a):
        out = []
        if CLICK in a.actions:
            colour_count = Counter(o.colour for o in a.objs)
            for o in sorted(a.objs, key=lambda o: (colour_count[o.colour], o.size)):
                r = a.mem.roles.get(o.type)
                if r and (r["deaths"] or (r["tries"] and not r["effects"])):
                    continue  # known deadly or inert: skip
                if (CLICK, o.type) in a.level_tried or a.tested(CLICK, o):
                    continue
                out.append((CLICK, o))
        # moves: probe until each move's effect is learned (then it's exploit territory)
        out += [(m, None) for m in a.actions if m != CLICK and m not in a.move_vec and not a.tested(m, None)]
        return out

    def applicable(self, a):
        return bool(self.candidates(a))

    def next(self, a):
        c = self.candidates(a)
        return c[0] if c else None


class Repeat(Skill):
    """Exploit: keep doing what just changed the world, until it stops changing (persistence)."""
    name, kind = "repeat", "exploit"

    def applicable(self, a):
        return a.last is not None and a.last_changed and a.last[0] != CLICK

    def start(self, a):
        self.act, self.n = a.last[0], 0

    def next(self, a):
        self.n += 1
        if self.n > 30 or (self.n > 1 and not a.last_changed):
            return None
        return (self.act, None)


class ClickAll(Skill):
    """Exploit: click every member of types that did something (collect / toggle-all template)."""
    name, kind = "click_all", "exploit"

    def todo(self, a):
        return [o for o in a.objs if a.mem.roles.get(o.type, {}).get("effects") and not a.mem.roles[o.type]["deaths"]
                and (o.x, o.y) not in a.level_clicked]

    def applicable(self, a):
        return CLICK in a.actions and bool(self.todo(a))

    def next(self, a):
        t = self.todo(a)
        return (CLICK, t[0]) if t else None


class GoTo(Skill):
    """Exploit: walk the player to each untried object type until touching (reach / touch template)."""
    name, kind = "go_to", "exploit"

    def targets(self, a):
        return [o for o in a.objs if o.type not in a.player and o.type not in a.level_reached
                and not a.mem.roles.get(o.type, {}).get("deaths") and o.size < a.grid.size // 8]

    def applicable(self, a):
        return bool(a.player) and bool(a.move_vec) and bool(self.targets(a))

    def start(self, a):
        self.stuck = 0
        self.target = None

    def next(self, a):
        p = a.player_pos()
        ts = self.targets(a)
        if p is None or not ts:
            return None
        if self.target not in {o.type for o in ts}:
            self.target = min(ts, key=lambda o: abs(o.cx - p[0]) + abs(o.cy - p[1])).type
        goal = min((o for o in ts if o.type == self.target), key=lambda o: abs(o.cx - p[0]) + abs(o.cy - p[1]))
        dist = abs(goal.cx - p[0]) + abs(goal.cy - p[1])
        self.stuck = 0 if a.last_changed else self.stuck + 1
        if dist <= 2 or self.stuck >= 3:
            a.level_reached.add(self.target)
            return None
        # pick the learned move that brings the player closest; skip moves that just failed in this state
        ok = [m for m in a.move_vec if not a.tested_noop(m)] or list(a.move_vec)
        m = min(ok, key=lambda m: abs(p[0] + a.move_vec[m][0] - goal.cx) + abs(p[1] + a.move_vec[m][1] - goal.cy))
        return (m, None)


class Replay(Skill):
    """Exploit: replay a winning recipe from memory on analogous objects."""
    name, kind = "replay", "exploit"

    def usable(self, a):
        for i, r in enumerate(a.mem.recipes):
            if i in a.level_failed_recipes:
                continue
            if all(t is None or a.mem.analog(t, a.objs) for _, t in r):
                return i
        return None

    def applicable(self, a):
        return self.usable(a) is not None

    def start(self, a):
        self.i, self.step = self.usable(a), 0

    def next(self, a):
        r = a.mem.recipes[self.i]
        if self.step >= len(r):
            a.level_failed_recipes.add(self.i)  # ran to the end without finishing the level
            return None
        act, t = r[self.step]
        self.step += 1
        if act not in a.actions:
            a.level_failed_recipes.add(self.i)
            return None
        if t is None:
            return (act, None)
        objs = a.mem.analog(t, a.objs)
        return (act, objs[0]) if objs else None


# ---------------------------------------------------------------- agent
class Agent:
    """Each step: finish the current skill episode, or pick a new skill (explore first, then best exploit)."""

    def __init__(self, memory=None, seed=0):
        self.mem = memory or Memory()
        self.rng = random.Random(seed)
        self.skills = [Probe(), Replay(), Repeat(), ClickAll(), GoTo()]
        self.usage = Counter()
        self.new_game()

    # ---- bookkeeping
    def new_game(self):
        self.player, self.move_vec, self.hud, self.hud_votes = set(), {}, None, None
        self.new_level()

    def new_level(self):
        self.tried = set()       # (state, action key) already done
        self.edges = {}          # (state, key) -> next state; for the shortest winning path
        self.start_state = None
        self.level_tried, self.level_reached, self.level_failed_recipes, self.level_clicked = set(), set(), set(), set()
        self.last, self.last_changed, self.skill = None, False, None

    def key(self, act, obj):
        return (act, obj.type if obj else None) if act == CLICK else (act, None)

    def tested(self, act, obj):
        return (self.state, self.key(act, obj)) in self.tried

    def tested_noop(self, act):
        k = (self.state, (act, None))
        return k in self.tried and self.edges.get(k) == self.state

    def player_pos(self):
        ps = [o for o in self.objs if o.type in self.player]
        if not ps:
            return None
        return (sum(o.cx for o in ps) / len(ps), sum(o.cy for o in ps) / len(ps))

    def perceive(self, f):
        self.grid = np.array(f.frame[-1])
        if self.hud is None or self.hud.shape != self.grid.shape:
            self.hud = np.zeros(self.grid.shape, bool)
            self.hud_votes = np.zeros(self.grid.shape, int)
        self.actions = [a.value if hasattr(a, "value") else int(a) for a in f.available_actions] or list(MOVES)
        self.objs = objects(self.grid, self.hud)
        masked = np.where(self.hud, -1, self.grid)
        self.state = hash(masked.tobytes())
        if self.start_state is None:
            self.start_state = self.state

    # ---- decide
    def value(self, s):
        v, n = self.mem.skill_stats.get(s.name, [1.0, 1])  # optimistic prior: every skill gets tried
        return v / n

    def choose(self, f):
        """Return (action id, {"x", "y"} | None)."""
        self.perceive(f)
        step = self.skill.next(self) if self.skill else None
        if step is None:
            self.skill = None
            explore = [s for s in self.skills if s.kind == "explore" and s.applicable(self)]
            exploit = [s for s in self.skills if s.kind == "exploit" and s.applicable(self)]
            best = max(exploit, key=self.value, default=None)
            replay = next((s for s in exploit if s.name == "replay"), None)
            # analogy first (cheap), then explore while unknown things remain, unless an exploit skill proved better
            if replay:
                self.skill = replay
            elif explore and (best is None or self.value(best) <= self.value(explore[0])):
                self.skill = explore[0]
            elif best:
                self.skill = best
            if self.skill:
                self.skill.start(self)
                step = self.skill.next(self)
        if step is None:  # nothing applicable: random fallback
            self.skill = None
            act = self.rng.choice(self.actions)
            step = (act, self.rng.choice(self.objs) if act == CLICK and self.objs else None)
        act, obj = step
        self.pending = (act, obj, self.skill.name if self.skill else "random")
        self.usage[self.pending[2]] += 1
        if act == CLICK:
            if obj is None:
                return act, {"x": self.rng.randrange(self.grid.shape[1]), "y": self.rng.randrange(self.grid.shape[0])}
            return act, {"x": obj.x, "y": obj.y}
        return act, None

    # ---- learn
    def update(self, f_prev, f_new):
        act, obj, skill = self.pending
        k = self.key(act, obj)
        before, s0 = self.objs, self.state
        new = np.array(f_new.frame[-1]) if f_new.frame else self.grid
        if new.shape == self.grid.shape:  # a band becomes HUD after changing on 2 separate steps
            self.hud_votes += hud_bands(self.grid, new, obj.box if obj else None)
            self.hud = self.hud_votes >= 2
        changed = new.shape != self.grid.shape or bool(((self.grid != new) & ~self.hud).any())
        level_up = f_new.levels_completed > f_prev.levels_completed
        died = f_new.state.name == "GAME_OVER"

        self.tried.add((s0, k))
        if act == CLICK and obj:
            self.level_tried.add(k)
            self.level_clicked.add((obj.x, obj.y))
            r = self.mem.role(obj.type)
            r["tries"] += 1
            r["effects"] += changed
            r["deaths"] += died

        # agentness: types that moved under a movement action are the player; learn each move's displacement
        if act in MOVES and changed and not level_up and new.shape == self.grid.shape:
            after = objects(new, self.hud)
            moved = self.displacements(before, after)
            if moved:
                self.player = set(moved) if not self.player else (self.player & set(moved)) or set(moved)
                dx = float(np.mean([d[0] for d in moved.values()]))
                dy = float(np.mean([d[1] for d in moved.values()]))
                self.move_vec[act] = (dx, dy)

        # skill credit: new state = small win, level = big win, death = loss
        seen = {v for v in self.edges.values()}
        gain = 10.0 * level_up + 1.0 * (changed and not died) - 5.0 * died
        if not died and not level_up:
            self.edges[(s0, k)] = hash(np.where(self.hud, -1, new).tobytes()) if new.shape == self.hud.shape else None
            gain = gain if self.edges[(s0, k)] not in seen else 0.2 * gain
        st = self.mem.skill_stats.setdefault(skill, [1.0, 1])
        st[0] += gain
        st[1] += 1

        self.last, self.last_changed = (act, obj), changed
        if level_up:
            self.store_recipe(s0, k)
            self.new_level()
        elif died:
            self.skill = None  # the attempt restarts; level knowledge (tried, edges) is kept

    def displacements(self, before, after):
        """type -> (dx, dy) for types whose single instance moved."""
        out = {}
        b = {}
        for o in before:
            b.setdefault(o.type, []).append(o)
        a = {}
        for o in after:
            a.setdefault(o.type, []).append(o)
        for t, bs in b.items():
            if len(bs) == 1 and len(a.get(t, [])) == 1:
                d = (a[t][0].cx - bs[0].cx, a[t][0].cy - bs[0].cy)
                if abs(d[0]) + abs(d[1]) >= 1:
                    out[t] = d
        return out

    def store_recipe(self, s_last, k_last):
        """Shortest known path start -> winning action, stored as (action, type) steps."""
        prev = {self.start_state: None}
        q = deque([self.start_state])
        while q:
            s = q.popleft()
            if s == s_last:
                break
            for (s0, k), s1 in self.edges.items():
                if s0 == s and s1 is not None and s1 not in prev:
                    prev[s1] = (s0, k)
                    q.append(s1)
        if s_last not in prev:
            return
        path, s = [k_last], s_last
        while prev[s]:
            s, k = prev[s]
            path.append(k)
        recipe = path[::-1]
        if recipe not in self.mem.recipes:
            self.mem.recipes.append(recipe)

    def on_reset(self):
        self.last, self.last_changed, self.skill = None, False, None


class RandomAgent:
    """Baseline: random available action, random pixel for clicks."""

    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.usage = Counter()

    def new_game(self):
        pass

    def on_reset(self):
        pass

    def choose(self, f):
        acts = [a.value if hasattr(a, "value") else int(a) for a in f.available_actions] or list(MOVES)
        act = self.rng.choice(acts)
        self.usage["random"] += 1
        g = f.frame[-1]
        return act, ({"x": self.rng.randrange(len(g[0])), "y": self.rng.randrange(len(g))} if act == CLICK else None)

    def update(self, f_prev, f_new):
        pass
