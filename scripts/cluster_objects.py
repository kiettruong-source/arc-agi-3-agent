import logging; logging.disable(logging.CRITICAL)
from collections import Counter
import numpy as np
from scipy.ndimage import label
import arc_agi

arc = arc_agi.Arcade()
print(f"{'game':6} {'acts':10} {'objects':>7} {'types':>5} {'bg':>3}")
for e in arc.get_environments():
    env = arc.make(e.game_id)
    f = env.reset()
    g = np.array(f.frame[-1])
    bg = Counter(g.ravel()).most_common(1)[0][0]
    objs, types = 0, set()
    for c in np.unique(g):
        if c == bg: continue
        lab, n = label(g == c)
        for i in range(1, n + 1):
            ys, xs = np.nonzero(lab == i)
            shape = (lab[ys.min():ys.max()+1, xs.min():xs.max()+1] == i).tobytes()
            objs += 1; types.add((int(c), shape))
    print(f"{e.game_id[:4]:6} {str(f.available_actions):10} {objs:7} {len(types):5} {bg:3}")
