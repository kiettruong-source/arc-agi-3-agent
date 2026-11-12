"""Minimal ARC-AGI-3 agent v1: perception -> fundamental skills -> explore / exploit, with an analogy memory.

The agent only uses what every game gives: a colour grid, the available actions, and level / game-over
signals. No game ids, no fixed grid size, no meaning attached to any colour.
"""

import hashlib
import random
from collections import Counter, deque
from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import label

import llm

MOVES = (1, 2, 3, 4)
CLICK = 6


def sid(arr):
    """Stable screen id (Python's hash() is randomised per process, which made runs irreproducible)."""
    return hashlib.blake2b(arr.tobytes(), digest_size=8).hexdigest()


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


def hud_bands(prev, new, exclude=()):
    """Thin changes hugging the border = counters / bars. Returns a mask of whole bands (a HUD vote).
    `exclude` = boxes (y0, x0, y1, x1) of things the agent acted on / moved: their change is an effect, not HUD."""
    mask = np.zeros(new.shape, bool)
    lab, n = label(prev != new, structure=np.ones((3, 3)))
    H, W = new.shape
    e = max(1, min(H, W) // 16)  # border margin scales with the grid (4 cells on 64x64)
    for i in range(1, n + 1):
        ys, xs = np.nonzero(lab == i)
        y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
        if any(not (y1 < b[0] or y0 > b[2] or x1 < b[1] or x0 > b[3]) for b in exclude):
            continue
        # ponytail: border+thin heuristic; a player walking along the edge could be masked. Learn per-cell stats if it bites.
        if y1 - y0 <= 2 and (y0 < e or y1 >= H - e):
            mask[y0:y1 + 1, :] = True
        elif x1 - x0 <= 2 and (x0 < e or x1 >= W - e):
            mask[:, x0:x1 + 1] = True
    return mask


# ---------------------------------------------------------------- memory
@dataclass
class Memory:
    """Analogy memory, shared across levels and games in one run."""
    roles: dict = field(default_factory=dict)    # type -> {"tries", "effects", "deaths"}
    recipes: list = field(default_factory=list)  # [[(action, target description | None), ...], ...]
    skill_stats: dict = field(default_factory=dict)  # name -> [value_sum, steps]
    goals: list = field(default_factory=list)    # [(schema, start features of the level it was learned on), ...]
    ops: dict = field(default_factory=dict)      # (action, type | None) -> [summed feature changes, tries]: what an action does

    def role(self, t):
        return self.roles.setdefault(t, {"tries": 0, "effects": 0, "deaths": 0})

    def known(self, t):
        return t in self.roles and self.roles[t]["tries"] > 0



# ---------------------------------------------------------------- structure mapping (analogy between levels / games)
def describe(o, objs, player_box=None):
    """An object's relational description: its role in the scene, not just its looks."""
    sizes = sorted({p.size for p in objs})
    rel = None
    if player_box:
        pcx, pcy = (player_box[1] + player_box[3]) / 2, (player_box[0] + player_box[2]) / 2
        rel = (int(np.sign(round(o.cx - pcx))), int(np.sign(round(o.cy - pcy))))
    return {
        "type": o.type, "colour": o.colour, "shape": o.type[1:3],                     # surface
        "odd": sum(p.colour == o.colour for p in objs) == 1,                          # relation: odd one out
        "same": sum(p.type == o.type for p in objs),                                  # relation: how many like it
        "rank": sizes.index(o.size) / max(1, len(sizes) - 1) if o.size in sizes else 0.5,  # relation: relative size
        "rel": rel,                                                                   # relation: side of the player
    }


def similarity(d, e):
    """Structure-mapping score in [0, 1]: relations weigh more than surface attributes (Gentner)."""
    s = 3.0 * (d["type"] == e["type"])                                  # literally the same kind of thing
    s += 2.0 * (d["odd"] == e["odd"])
    s += 1.0 * min(d["same"], e["same"]) / max(d["same"], e["same"], 1)
    s += 1.0 * (1 - abs(d["rank"] - e["rank"]))
    s += 0.5 * (d["shape"] == e["shape"]) + 0.5 * (d["colour"] == e["colour"])
    total = 8.0
    if d["rel"] is not None and e["rel"] is not None:
        s += 1.0 * (d["rel"] == e["rel"])
        total += 1.0
    return s / total


def best_match(desc, objs, player_box=None, used=(), threshold=0.5):
    """The object in the new scene that plays the same role, or None if nothing maps well enough."""
    scored = [(similarity(desc, describe(o, objs, player_box)), o) for o in objs if (o.x, o.y) not in used]
    scored = [x for x in scored if x[0] >= threshold]
    return max(scored, key=lambda x: x[0])[1] if scored else None


# ---------------------------------------------------------------- goal reasoning: event -> contrast -> hypothesis
# General by construction: a situation is described by property (colour, size class) x measure (how many,
# how many cells, distance to what I control) plus global counts. No goal types are written by hand; the goal
# is whatever distinguishes the winning screen from every screen where the level did not end.
def size_class(n):
    return 0 if n <= 30 else 1 if n <= 120 else 2


def gap(a, b):
    """Empty cells between two boxes (y0, x0, y1, x1); 0 = touching or overlapping."""
    return max(b[1] - a[3] - 1, a[1] - b[3] - 1, 0) + max(b[0] - a[2] - 1, a[0] - b[2] - 1, 0)


def features(objs, player=(), step=1):
    """Situation features. `player` = types of the thing the agent controls (may be empty)."""
    f = {"objects": len(objs), "types": len({o.type for o in objs})}
    by = {}
    for o in objs:
        by.setdefault(o.colour, []).append(o)
        k = ("count", "size", size_class(o.size))
        f[k] = f.get(k, 0) + 1
    f["colours"] = len(by)
    f["odd"] = sum(len(v) == 1 for v in by.values())
    me = [o.box for o in objs if o.type in player]
    box = (min(b[0] for b in me), min(b[1] for b in me), max(b[2] for b in me), max(b[3] for b in me)) if me else None
    for c, group in by.items():
        f[("count", "colour", c)] = len(group)
        f[("cells", "colour", c)] = sum(o.size for o in group)
        others = [o for o in group if o.type not in player]
        if box and others:
            f[("dist", "colour", c)] = round(min(gap(box, o.box) for o in others) / max(step, 1))
    return f


def val(f, k):
    """A missing count is 0; a missing distance is unknown (the thing is not there)."""
    return f.get(k, None if isinstance(k, tuple) and k[0] == "dist" else 0)


def induce(win, others, start):
    """Contrast (method of difference): features whose value on the winning screen was never seen on a
    non-winning screen (weight 2) or at least differs from the level's start (weight 1).
    Returns {feature: (direction, to_zero, magnitude, weight)}."""
    schema = {}
    for k in set(win) | set(start):
        w, s0 = val(win, k), val(start, k)
        if w is None or s0 is None or w == s0:
            continue
        seen = [v for v in (val(o, k) for o in others) if v is not None]
        novel = bool(seen) and (w < min(seen) or w > max(seen))
        schema[k] = (1 if w > s0 else -1, w == 0, abs(w - s0), 2 if novel else 1)
    return schema


def translate(schema, src_start, dst_start):
    """Carry a schema to another level: a colour that is absent there is replaced by the colour of the same
    rarity rank (structure mapping at feature level; also makes the reasoning colour-permutation safe)."""
    def ranked(f):
        cs = [k[2] for k in f if isinstance(k, tuple) and k[:2] == ("count", "colour")]
        return sorted(cs, key=lambda c: (f[("count", "colour", c)], f.get(("cells", "colour", c), 0), c))
    src, dst = ranked(src_start), ranked(dst_start)
    out = {}
    for k, v in schema.items():
        if isinstance(k, tuple) and k[1] == "colour" and k[2] not in dst:
            # ponytail: rank mapping only when both scenes have the same number of colours; map by role if it bites
            if len(src) != len(dst) or k[2] not in src:
                continue
            k = (k[0], "colour", dst[src.index(k[2])])
        out[k] = v
    return out


def progress(f, schema, start):
    """How far a situation has moved from this level's start in the schema's directions, in [0, 1]."""
    num = den = 0.0
    for k, (d, to_zero, mag, w) in schema.items():
        v, s0 = val(f, k), val(start, k)
        span = s0 if to_zero else mag  # "to zero": all of it has to go; otherwise: as far as it went last time
        if v is None or s0 is None or span <= 0:
            continue
        num += w * min(1.0, max(0.0, d * (v - s0) / span))
        den += w
    return num / den if den else 0.0


# A goal hypothesis from outside (the LLM) is a list of conditions over the same features. It is data, not code:
# the core can check it against the evidence and pursue it with the same machinery as a goal learned from a win.
OPS = {"==": lambda a, b: a == b, "<=": lambda a, b: a <= b, ">=": lambda a, b: a >= b}


def conditions(goal):
    """Validate {"all": [{"measure", "colour" | "size", "op", "value"}, ...]} -> [(feature, op, value)] or None."""
    out = []
    for c in (goal.get("all") or []) if isinstance(goal, dict) else []:
        if not isinstance(c, dict):
            return None
        m, op, v = c.get("measure"), c.get("op"), c.get("value")
        if op not in OPS or isinstance(v, bool) or not isinstance(v, (int, float)):
            return None
        if m in ("objects", "types", "colours", "odd"):
            key = m
        elif m in ("count", "cells", "dist") and isinstance(c.get("colour"), int):
            key = (m, "colour", c["colour"])
        elif m == "count" and c.get("size") in (0, 1, 2):
            key = ("count", "size", c["size"])
        else:
            return None
        out.append((key, op, v))
    return out or None


def holds(conds, f):
    return all(val(f, k) is not None and OPS[op](val(f, k), x) for k, op, x in conds)


def to_schema(conds, start):
    """A proposed goal in the form the search pursues; parts already true at the level's start drop out."""
    schema = {}
    for k, op, x in conds:
        s0 = val(start, k)
        if s0 is not None and not OPS[op](s0, x):
            schema[k] = (1 if x > s0 else -1, x == 0, abs(x - s0), 2)
    return schema


def schema_text(schema, top=4):
    """Readable form for logs and reports."""
    items = sorted(schema.items(), key=lambda kv: -kv[1][3])[:top]
    name = lambda k: " ".join(str(x) for x in k) if isinstance(k, tuple) else k
    return [f"{name(k)} {'to 0' if z else ('up' if d > 0 else 'down')}{' (never seen before)' if w == 2 else ''}"
            for k, (d, z, m, w) in items]


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
                if a.suspect_hud(o):
                    continue
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
        return [o for o in a.objs if not a.suspect_hud(o) and a.mem.roles.get(o.type, {}).get("effects") and not a.mem.roles[o.type]["deaths"]
                and (o.x, o.y) not in a.level_clicked]

    def applicable(self, a):
        return CLICK in a.actions and bool(self.todo(a))

    def next(self, a):
        t = self.todo(a)
        return (CLICK, t[0]) if t else None


class GoTo(Skill):
    """Exploit: plan a route on the mental map to the nearest untried object type, walk it (reach / touch template)."""
    name, kind = "go_to", "exploit"

    def targets(self, a):
        return [o for o in a.objs if o.type not in a.player and o.type not in a.level_reached
                and not a.mem.roles.get(o.type, {}).get("deaths") and o.size < a.grid.size // 8]

    def goal_targets(self, a):
        """Means-ends for movement: the goal says some colour's distance / count must go down -> walk there."""
        goal = a.active_goal()
        if not goal:
            return []
        cols = {k[2] for k, v in goal.items() if isinstance(k, tuple) and k[1] == "colour" and v[0] < 0}
        return [o for o in self.targets(a) if o.colour in cols]

    def applicable(self, a):
        return a.player_box() is not None and bool(a.move_vec) and bool(self.targets(a))

    def next(self, a):
        ts = self.goal_targets(a) or self.targets(a)
        if not ts or a.player_box() is None:
            return None
        hit = [o for o in ts if a.touching(o)]
        if hit:  # arrived: this goal guess has been tried
            a.level_reached.update(o.type for o in hit)
            return None
        m = a.plan(lambda box: any(a.touching(o, box) for o in ts), allow=ts)
        if m is None:  # nothing reachable on the current map
            a.level_reached.update(o.type for o in ts)
            return None
        a.nav_target = ts
        return (m, None)


class ExploreMap(Skill):
    """Explore places: plan a route to the nearest reachable spot the player has never stood on."""
    name, kind = "explore_map", "explore"

    def applicable(self, a):
        return a.player_box() is not None and bool(a.move_vec) and not GoTo().applicable(a)

    def next(self, a):
        if a.player_box() is None:
            return None
        m = a.plan(lambda box: box[:2] not in a.visited)
        return (m, None) if m is not None else None


class Replay(Skill):
    """Exploit: replay a winning recipe from memory on analogous objects."""
    name, kind = "replay", "exploit"

    def usable(self, a):
        for i, r in enumerate(a.mem.recipes):
            if i in a.level_failed_recipes or ("replay5" not in a.off and len(r) > 5):
                continue  # long action sequences do not fit a new layout (0 wins in 9 tries, logbook F10)
            if all(d is None or best_match(d, a.objs, a.player_box()) for _, d in r):
                return i
        return None

    def applicable(self, a):
        return self.usable(a) is not None

    def start(self, a):
        self.i, self.step, self.used = self.usable(a), 0, set()

    def next(self, a):
        r = a.mem.recipes[self.i]
        if self.step >= len(r):
            a.level_failed_recipes.add(self.i)  # ran to the end without finishing the level
            return None
        act, d = r[self.step]
        self.step += 1
        if act not in a.actions:
            a.level_failed_recipes.add(self.i)
            return None
        if d is None:
            return (act, None)
        o = best_match(d, a.objs, a.player_box(), self.used)  # one-to-one mapping within a replay
        if o is None:
            a.level_failed_recipes.add(self.i)
            return None
        self.used.add((o.x, o.y))
        return (act, o)


class Frontier(Skill):
    """Graph exploration (arXiv 2512.24156): every screen is a node, every action tried there an edge.
    Do an untried action here; otherwise walk the shortest known path to the nearest screen that still has one.
    Small objects / plain actions first (tier 0); larger tiers are allowed only when nothing smaller is left."""
    name, kind = "frontier", "explore"

    def untested(self, a, s, p):
        return [(k, typ) for t, k, typ in a.keys_at.get(s, []) if t <= p and (s, k) not in a.tried_exec]

    def next(self, a):
        adj = {}
        for (s0, k), s1 in a.exec_edges.items():
            if s1 is not None and s1 != s0:
                adj.setdefault(s0, []).append((k, s1))
        goal = a.active_goal()
        for p in range(4):  # raise the tier only when nothing of the current tier is reachable
            first, dist, cands = {a.state: None}, {a.state: 0}, []
            q = deque([a.state])
            while q:  # every reachable screen that still has an untried action
                s = q.popleft()
                if self.untested(a, s, p):
                    cands.append(s)
                    if not goal:
                        break  # no goal: the nearest one (v4 behaviour)
                for k, s1 in adj.get(s, []):
                    if s1 not in first:
                        first[s1], dist[s1] = first[s] or k, dist[s] + 1
                        q.append(s1)
            if not cands:
                continue
            # with a goal: the screen closest to the goal, minus the cost of walking there (best-first)
            best = max(cands, key=lambda s: a.progress_of(s) - a.step_cost() * dist[s]) if goal else cands[0]
            if best != a.state:
                return a.exec_key(first[best])
            here = self.untested(a, a.state, p)
            if goal:  # means-ends: the action whose known effect shrinks the difference to the goal comes first
                here.sort(key=lambda kt: -a.gain(kt[0][0], kt[1]))
            return a.exec_key(here[0][0])
        return None


# ---------------------------------------------------------------- agent
class Agent:
    """Each step: finish the current skill episode, or pick a new skill (explore first, then best exploit)."""

    # Parts that did not help on the dev games stay off unless asked for (protocol E12): the deadline rule and
    # short replay each traded one fragile level for another. Goal use is on: neutral on dev, exact in toy worlds.
    DEFAULT_OFF = ("deadline", "replay5")

    def __init__(self, memory=None, seed=0, off=None):
        self.off = set(self.DEFAULT_OFF if off is None else off)  # switches: "goal", "deadline", "replay5", "llm"
        self.mem = memory or Memory()
        self.rng = random.Random(seed)
        self.skills = [Probe(), ExploreMap(), Replay(), Repeat(), ClickAll(), GoTo()]
        self.usage = Counter()
        self.goal_log = []  # one entry per completed level: what was learned, and how well the old goal predicted it
        self.llm_log = []   # one entry per question to the LLM: how many goals it proposed, how many survived the evidence
        self.new_game()

    # ---- bookkeeping
    def new_game(self):
        self.player, self.move_vec, self.hud, self.hud_votes = set(), {}, None, None
        self.floor = set()       # colours the player has walked onto (mental map: free space)
        self.goal, self.level = None, 0   # goal schema learned in this game (schema, source start features)
        self.new_level()

    def new_level(self):
        self.tried = set()       # (state, action key) already done
        self.tried_exec, self.exec_edges, self.keys_at = set(), {}, {}  # state graph for frontier exploration
        self.edges = {}          # (state, key) -> next state; for the shortest winning path
        self.start_state = None
        self.level_tried, self.level_reached, self.level_failed_recipes, self.level_clicked = set(), set(), set(), set()
        self.last, self.last_changed, self.skill = None, False, None
        self.visited, self.blocked, self.nav_target = set(), set(), None  # mental map of this level
        self.desc = {}  # type -> relational description when acted on (for structure mapping)
        self.seen = {}  # state -> grid: every screen of this level where the level did NOT end (free evidence)
        self.start_grid = None
        self.goal_now, self.start_feat, self.fcache, self.feats = None, None, {}, {}  # goal carried into this level
        self.best_progress, self.since_progress, self.stalled = 0.0, 0, False
        self.attempt_steps, self.budget = 0, None  # actions since the level (re)started; step limit seen at a game over
        self.hyps, self.goal_conds, self.goal_why, self.refuted = [], None, None, []  # proposed goals waiting / in test
        self.level_steps, self.asks, self.last_ask = 0, 0, -999

    def key(self, act, obj):
        return (act, obj.type if obj else None) if act == CLICK else (act, None)

    def tested(self, act, obj):
        return (self.state, self.key(act, obj)) in self.tried

    def exec_key(self, k):
        """Graph action key -> (action, object | None) on the current screen."""
        if k[0] == CLICK:
            o = next((o for o in self.objs if (o.x, o.y) == (k[1], k[2])), None)
            return (CLICK, o) if o else None
        return (k[0], None)

    def suspect_hud(self, o):
        """Object inside a border band that already changed on its own once: probably a counter, don't touch."""
        return self.hud_votes is not None and bool(self.hud_votes[o.box[0]:o.box[2] + 1, o.box[1]:o.box[3] + 1].any())

    def player_box(self, objs=None):
        """Union box (y0, x0, y1, x1) of the player's parts."""
        ps = [o for o in (self.objs if objs is None else objs) if o.type in self.player]
        if not ps:
            return None
        return (min(o.box[0] for o in ps), min(o.box[1] for o in ps), max(o.box[2] for o in ps), max(o.box[3] for o in ps))

    def steps(self):
        return {m: (round(dx), round(dy)) for m, (dx, dy) in self.move_vec.items() if round(dx) or round(dy)}

    def touching(self, o, box=None):
        """Player box (grown by one step) overlaps the object."""
        b = box or self.player_box()
        r = max([max(abs(dx), abs(dy)) for dx, dy in self.steps().values()] or [1])
        return not (b[2] + r < o.box[0] or b[0] - r > o.box[2] or b[3] + r < o.box[1] or b[1] - r > o.box[3])

    def plan(self, goal, allow=()):
        """Shortest route on the mental map (BFS over player positions). Returns the first move, or None.
        Free = floor colours + the player's own colours (+ cells of `allow` objects); known blocked moves are walls."""
        start = self.player_box()
        H, W = self.grid.shape
        pcols = {o.colour for o in self.objs if o.type in self.player}
        free = np.isin(self.grid, list(self.floor | pcols)) & ~self.hud
        for o in allow:
            free[o.box[0]:o.box[2] + 1, o.box[1]:o.box[3] + 1] = True
        steps = self.steps()
        first = {start: None}
        q = deque([start])
        while q and len(first) < 20000:  # ponytail: plain BFS, A* if grids get big
            b = q.popleft()
            for m, (dx, dy) in steps.items():
                if (b[:2], m) in self.blocked:
                    continue
                nb = (b[0] + dy, b[1] + dx, b[2] + dy, b[3] + dx)
                if nb in first or nb[0] < 0 or nb[1] < 0 or nb[2] >= H or nb[3] >= W:
                    continue
                if not free[nb[0]:nb[2] + 1, nb[1]:nb[3] + 1].all():
                    continue
                first[nb] = first[b] or m
                if goal(nb):
                    return first[nb]
                q.append(nb)
        return None

    def perceive(self, f):
        self.grid = np.array(f.frame[-1])
        if self.hud is None or self.hud.shape != self.grid.shape:
            self.hud = np.zeros(self.grid.shape, bool)
            self.hud_votes = np.zeros(self.grid.shape, int)
        self.actions = [a.value if hasattr(a, "value") else int(a) for a in f.available_actions] or list(MOVES)
        self.objs = objects(self.grid, self.hud)
        masked = np.where(self.hud, -1, self.grid)
        self.state = sid(masked)
        if self.state not in self.keys_at:  # executable actions on this screen, with a salience tier
            keys = [(0, (m, None), None) for m in self.actions if m != CLICK]
            if CLICK in self.actions:
                for o in self.objs:
                    if not self.suspect_hud(o):
                        keys.append((0 if o.size <= 30 else 1 if o.size <= 120 else 2 if o.size <= 500 else 3,
                                     (CLICK, o.x, o.y), o.type))
            self.keys_at[self.state] = sorted(keys, key=lambda x: x[0])
        if self.start_state is None:
            self.start_state, self.start_grid = self.state, self.grid
            if self.goal and "goal" not in self.off:  # carry the goal (not the actions) into the new level
                self.start_feat = self.feat(self.grid)
                self.goal_now = translate(self.goal[0], self.goal[1], self.start_feat) or None
        self.seen.setdefault(self.state, self.grid)
        if self.goal_conds and holds(self.goal_conds, self.feat_of(self.state)):
            # the proposed goal is true on this screen and the level did not end: refuted by a counter-example
            self.refuted.append(self.goal_why)
            self.next_hypothesis()
        if self.goal_now and not self.stalled:  # is the goal in test getting us anywhere?
            pr = self.progress_of(self.state)
            if pr > self.best_progress + 1e-9:
                self.best_progress, self.since_progress = pr, 0
            else:
                self.since_progress += 1
                if self.since_progress > (self.budget // 2 if self.budget else 30):
                    self.next_hypothesis()  # give up on it: next proposal, or back to plain exploring
        self.maybe_ask()
        b = self.player_box() if self.player else None
        if b:
            self.visited.add(b[:2])

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
            goto = next(s for s in self.skills if s.name == "go_to")
            # a carried goal comes first (reverse thinking: from the goal to the action); then analogy (cheap),
            # then explore while unknown things remain, unless an exploit skill proved better
            if self.active_goal():
                self.skill = goto if goto.applicable(self) and goto.goal_targets(self) else Frontier()
            elif replay:
                self.skill = replay
            elif explore and (best is None or self.value(best) <= self.value(explore[0])):
                self.skill = explore[0]
            elif best:
                self.skill = best
            if self.skill:
                self.skill.start(self)
                step = self.skill.next(self)
        if step is None:  # nothing applicable: graph exploration, then random
            self.skill = Frontier()
            step = self.skill.next(self)
        if step is None:
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
        changed = new.shape != self.grid.shape or bool(((self.grid != new) & ~self.hud).any())
        level_up = f_new.levels_completed > f_prev.levels_completed
        died = f_new.state.name == "GAME_OVER"
        # deadline reasoning (E10): a game over at the step limit is a timeout, nobody's fault. Only a death
        # clearly before a limit already seen is blamed on what was just touched; the first one never is.
        self.attempt_steps += 1
        self.level_steps += 1
        blame = died and ("deadline" in self.off or (self.budget is not None and self.attempt_steps < 0.8 * self.budget))
        if died:
            self.budget = max(self.budget or 0, self.attempt_steps)

        self.tried.add((s0, k))
        ek = (CLICK, obj.x, obj.y) if act == CLICK and obj else (act, None)
        self.tried_exec.add((s0, ek))
        if act == CLICK and obj:
            self.level_tried.add(k)
            self.level_clicked.add((obj.x, obj.y))
            self.desc[obj.type] = describe(obj, before, self.player_box(before) if self.player else None)
            r = self.mem.role(obj.type)
            r["tries"] += 1
            r["effects"] += changed
            r["deaths"] += blame

        # agentness: types that moved under a movement action are the player; learn each move's displacement
        if act != CLICK and not level_up and new.shape == self.grid.shape:
            b0 = self.player_box(before) if self.player else None
            after = objects(new, self.hud)
            moved = self.displacements(before, after) if changed else {}
            if moved:
                self.player = set(moved) if not self.player else (self.player & set(moved)) or set(moved)
                dx = float(np.mean([d[0] for d in moved.values()]))
                dy = float(np.mean([d[1] for d in moved.values()]))
                self.move_vec[act] = (dx, dy)
            # mental map: where the player stepped is floor; a move that didn't move it (or killed it) is a wall
            b1 = self.player_box(after) if self.player else None
            if b0 and (b1 == b0 or blame):
                self.blocked.add((b0[:2], act))
            elif b0 and b1:
                region = self.grid[b1[0]:b1[2] + 1, b1[1]:b1[3] + 1]
                pcols = {o.colour for o in before if o.type in self.player}
                self.floor |= {int(c) for c in np.unique(region)} - pcols
        if new.shape == self.grid.shape:  # a band becomes HUD after changing on 2 separate steps
            boxes = [obj.box] if obj and not self.suspect_hud(obj) else []
            if act != CLICK and self.player:
                boxes += [b for b in (self.player_box(before), self.player_box(objects(new, self.hud))) if b]
            self.hud_votes += hud_bands(self.grid, new, boxes)
            self.hud = self.hud_votes >= 2

        # skill credit: new state = small win, level = big win, death = loss
        seen = {v for v in self.edges.values()}
        gain = 10.0 * level_up + 1.0 * (changed and not died) - 5.0 * blame
        if not died and not level_up:
            s1 = sid(np.where(self.hud, -1, new)) if new.shape == self.hud.shape else None
            self.edges[(s0, k)] = self.exec_edges[(s0, ek)] = s1
            gain = gain if s1 not in seen else 0.2 * gain
            if s1 is not None:  # operator table: what does this kind of action change? (means-ends analysis)
                fa, fb = self.feat_of(s0), self.feat(new)
                op = self.mem.ops.setdefault((act, obj.type if act == CLICK and obj else None), [{}, 0])
                op[1] += 1
                for key in set(fa) | set(fb):
                    va, vb = val(fa, key), val(fb, key)
                    if va is not None and vb is not None and va != vb:
                        op[0][key] = op[0].get(key, 0) + vb - va
        st = self.mem.skill_stats.setdefault(skill, [1.0, 1])
        st[0] += gain
        st[1] += 1

        self.last, self.last_changed = (act, obj), changed
        if level_up:
            self.store_recipe(s0, k)
            self.learn_goal(f_new)
            self.level += 1
            self.new_level()
        elif died:
            self.skill = None  # the attempt restarts; level knowledge (tried, edges) is kept

    def active_goal(self):
        return self.goal_now if self.goal_now and not self.stalled else None

    def next_hypothesis(self):
        """Drop the goal in test; take the next proposal, or fall back to plain exploring."""
        self.skill, self.fcache, self.best_progress, self.since_progress = None, {}, 0.0, 0
        if self.hyps:
            self.goal_why, self.goal_conds, self.goal_now = self.hyps.pop(0)
        else:
            self.goal_why, self.goal_conds, self.goal_now, self.stalled = None, None, None, True

    def maybe_ask(self):
        """Ask the LLM for goal hypotheses when the agent has looked around and has no goal to pursue.
        Its answers are never trusted: anything already true on a non-winning screen is thrown out."""
        if "llm" in self.off or not llm.enabled() or self.asks >= 2 or self.active_goal():
            return
        if self.level_steps < 12 or self.level_steps - self.last_ask < 60:
            return
        self.asks, self.last_ask = self.asks + 1, self.level_steps
        start = self.feat(self.start_grid)
        seen = [self.feat_of(s) for s in self.seen]
        proposals = llm.propose(self, start, seen) or []
        entry = {"level": self.level, "step": self.level_steps, "proposed": len(proposals), "kept": 0, "why": []}
        self.llm_log.append(entry)
        for why, conds in proposals:
            schema = to_schema(conds, start)
            if schema and not any(holds(conds, f) for f in seen):
                self.hyps.append((why, conds, schema))
                entry["kept"] += 1
                entry["why"].append(why)
        if self.hyps:
            self.start_feat, self.stalled = start, False
            self.next_hypothesis()

    def feat_of(self, s):
        if s not in self.feats:
            self.feats[s] = self.feat(self.seen[s]) if s in self.seen else {}
        return self.feats[s]

    def progress_of(self, s):
        if s not in self.fcache:
            self.fcache[s] = progress(self.feat_of(s), self.goal_now, self.start_feat)
        return self.fcache[s]

    def step_cost(self):
        """Walking is paid in progress: a full attempt's worth of steps costs as much as reaching the goal."""
        return 1.0 / self.budget if self.budget else 0.02

    def gain(self, act, typ):
        """Predicted progress from one action of this kind, from the operator table (0 if unknown)."""
        op = self.mem.ops.get((act, typ))
        if not op or not op[1]:
            return 0.0
        g = 0.0
        for k, (d, to_zero, mag, w) in self.goal_now.items():
            span = val(self.start_feat, k) if to_zero else mag
            if k in op[0] and span:
                g += w * d * (op[0][k] / op[1]) / span
        return g

    def feat(self, grid):
        step = max([max(abs(dx), abs(dy)) for dx, dy in self.steps().values()] or [1])
        return features(objects(grid, self.hud), self.player, step)

    def learn_goal(self, f_new):
        """At a win: contrast the winning screen (second-to-last frame, E10) with all non-winning screens."""
        entry = {"level": self.level, "screens": len(self.seen), "schema": None, "rank": None,
                 "llm_goal": self.goal_why, "llm_goal_true_at_win": None, "llm_goals_refuted": len(self.refuted)}
        self.goal_log.append(entry)
        frames = f_new.frame or []
        # the last level has no "next level" frame: its final frame is the winning screen
        shot = frames[-1:] if f_new.state.name == "WIN" else frames[-2:-1]
        if not shot or np.array(shot[0]).shape != self.hud.shape:
            return  # the winning screen was not shown: nothing to contrast
        win = self.feat(np.array(shot[0]))
        others = [self.feat(g) for g in self.seen.values()]
        start = self.feat(self.start_grid)
        if self.goal_conds:  # was the LLM's goal in test actually true on the winning screen?
            entry["llm_goal_true_at_win"] = holds(self.goal_conds, win)
        if self.goal:  # measurement: would the goal learned earlier have picked out this winning screen?
            sch = translate(self.goal[0], self.goal[1], start)
            pw = progress(win, sch, start)
            entry["rank"] = sum(progress(o, sch, start) >= pw for o in others) / max(1, len(others))
        schema = induce(win, others, start)
        if schema:
            self.goal = (schema, start)
            self.mem.goals.append(self.goal)
            entry["schema"] = schema_text(schema)

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
        recipe = [(act, self.desc.get(t, {"type": t, "colour": t[0], "shape": t[1:3], "odd": False, "same": 1,
                                         "rank": 0.5, "rel": None}) if t else None) for act, t in path[::-1]]
        if recipe not in self.mem.recipes:
            self.mem.recipes.append(recipe)

    def on_reset(self):
        self.attempt_steps = 0
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
