import logging, sys; logging.disable(logging.CRITICAL)
from collections import Counter
import numpy as np
from scipy.ndimage import label
import arc_agi
from arcengine import GameAction

TYPES = {}  # (colour, shape bytes) -> letter ; same letter = same type, anywhere

def objects(g):
    bg = Counter(g.ravel()).most_common(1)[0][0]
    out = []
    for c in np.unique(g):
        if c == bg: continue
        lab, n = label(g == c)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            m = lab[ys.min():ys.max()+1, xs.min():xs.max()+1] == i
            key = (int(c), m.shape, m.tobytes())          # position-free identity
            t = TYPES.setdefault(key, chr(65 + len(TYPES)))
            out.append((t, int(c), int(ys.min()), int(xs.min()), m.shape, len(ys)))
    return out

def show(objs):
    grid = [["." for _ in range(32)] for _ in range(32)]   # 2x downscale
    for t, c, y, x, (h, w), n in objs:
        for yy in range(y, y + h, 2):
            for xx in range(x, x + w, 2):
                grid[yy // 2][xx // 2] = t
    print("\n".join("".join(r) for r in grid))

game = sys.argv[1] if len(sys.argv) > 1 else "ls20"
act = sys.argv[2] if len(sys.argv) > 2 else "ACTION1"
arc = arc_agi.Arcade(); env = arc.make(game)
a = objects(np.array(env.reset().frame[-1]))
print(f"{game}: {len(a)} objects, {len(TYPES)} types\n")
show(a)
print("\ntype colour  y,x   size  cells   (count)")
cnt = Counter(o[0] for o in a)
for t in sorted(cnt):
    o = next(o for o in a if o[0] == t)
    print(f"  {t}   {o[1]:3}  {o[2]:2},{o[3]:2}  {o[4]}  {o[5]:4}   x{cnt[t]}")

b = objects(np.array(env.step(GameAction[act]).frame[-1]))
sa, sb = Counter((o[0], o[2], o[3]) for o in a), Counter((o[0], o[2], o[3]) for o in b)
gone, new = sa - sb, sb - sa
print(f"\nafter {act}:")
for t, y, x in gone:
    m = [n for n in new if n[0] == t]
    print(f"  type {t} at {y},{x} -> " + (f"moved to {m[0][1]},{m[0][2]}" if m else "disappeared/changed"))
for t, y, x in new:
    if not any(g[0] == t for g in gone): print(f"  new type {t} at {y},{x}")
if not gone and not new: print("  nothing changed")
