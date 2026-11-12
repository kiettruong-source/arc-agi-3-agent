# How well does an agent gather information? same observer for every agent
import os
import sys
import logging

if os.environ.get("PYTHONHASHSEED") != "0":  # fixed hash seed -> identical runs (set order of byte-keyed types)
    os.execve(sys.executable, [sys.executable] + sys.argv, {**os.environ, "PYTHONHASHSEED": "0"})

logging.disable(logging.CRITICAL)
sys.path.insert(0, ".")
import numpy as np, arc_agi
from arcengine import GameAction
from agent import Agent, RandomAgent, objects, hud_bands, sid
from run import DEV

B = 300
def hit(objs, x, y):
    for o in objs:
        y0, x0, y1, x1 = o.box
        if y0 <= y <= y1 and x0 <= x <= x1: return o
    return None

def measure(game, agent):
    arc = arc_agi.Arcade(); env = arc.make(game); f = env.reset(); agent.new_game()
    g = np.array(f.frame[-1]); votes = np.zeros(g.shape, int); hud = votes >= 2
    init_types = {o.type for o in objects(g, hud)}
    click = 6 in [a.value if hasattr(a, "value") else int(a) for a in f.available_actions]
    seen, tried, probed = set(), set(), set()
    eff = redundant = 0; cover = {}; t_all = None
    for step in range(1, B + 1):
        if f.state.name in ("WIN",): break
        if f.state.name == "GAME_OVER":
            f = env.reset(); agent.on_reset(); g = np.array(f.frame[-1]); continue
        objs = objects(g, hud)
        s = sid(np.where(hud, -1, g)); seen.add(s)
        act, data = agent.choose(f)
        if act == 0:  # deliberate level reset
            f = env.reset(); agent.on_reset(); g = np.array(f.frame[-1]); continue
        o = hit(objs, data["x"], data["y"]) if data else None
        k = (act, o.type if o else None)
        redundant += (s, k) in tried; tried.add((s, k))
        if o: probed.add(o.type)
        f2 = env.step(GameAction[f"ACTION{act}"], data=data)
        if f2 is None: break
        agent.update(f, f2)
        g2 = np.array(f2.frame[-1])
        if g2.shape == g.shape:
            votes += hud_bands(g, g2, [o.box] if o else []); hud = votes >= 2
            eff += bool(((g != g2) & ~hud).any())
        f, g = f2, g2
        if click and init_types:
            c = len(probed & init_types) / len(init_types)
            if step in (50, 150, 300): cover[step] = c
            if t_all is None and c == 1: t_all = step
    seen.add(sid(np.where(hud, -1, g)))
    return dict(eff=eff / B, states=len(seen), red=redundant / B, click=click, cover=cover, t_all=t_all)

rows = {n: {} for n in ("random", "v1")}
for game in DEV:
    rows["random"][game] = measure(game, RandomAgent(0))
    rows["v1"][game] = measure(game, Agent())
print(f"{'game':5} {'type':5} | {'useful%':>14} | {'new screens':>12} | {'wasted%':>12} | {'cover@50/150/300 (click)':>36} | {'all types by':>13}")
for game in DEV:
    r, v = rows["random"][game], rows["v1"][game]
    cv = lambda m: "/".join(f"{m['cover'].get(s,0):.0%}" for s in (50,150,300)) if m["click"] else "-"
    print(f"{game:5} {'click' if r['click'] else 'move':5} | {r['eff']:5.0%} -> {v['eff']:5.0%} | {r['states']:4} -> {v['states']:4} | {r['red']:4.0%} -> {v['red']:4.0%} | {cv(r):>16} -> {cv(v):>16} | {str(r['t_all']):>5} -> {str(v['t_all']):>5}")
for n in rows:
    m = rows[n].values(); cl = [x for x in m if x["click"]]
    print(f"\n{n:7} useful {np.mean([x['eff'] for x in m]):.0%}  screens {sum(x['states'] for x in m)}  wasted {np.mean([x['red'] for x in m]):.0%}  click-cover@300 {np.mean([x['cover'].get(300,0) for x in cl]):.0%}  all-types-reached {sum(x['t_all'] is not None for x in cl)}/{len(cl)}")
