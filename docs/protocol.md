# Experiment protocol — ARC-AGI-3 agent

Project: an agent for ARC-AGI-3 interactive games that discovers each game's rules and **goal** through cheap
exploration, generalises to unseen environments, and is tested by the course instructor on environments not known in
advance. Background research, sources and design notes: [research/2026-11-04-exploration-notes.md](research/2026-11-04-exploration-notes.md).
Raw history behind this record (process journal, every idea, every failure): [logbook.md](logbook.md). Evidence files: `runs/`.

## 1. Research question

How can an agent find out **what an unknown game wants** (its goal / end condition) using as few actions as
possible, by gathering information cheaply, giving what it sees meaning, and applying common-sense priors?

## 1b. Goal: an agent with the key characteristics of AGI

The agent should show, within the ARC-AGI-3 setting, the key characteristics of artificial general intelligence:

- **Generalization:** it can learn and adapt to entirely new, unfamiliar domains.
- **Cross-domain transfer:** it can transfer skills and abstract knowledge from one area to another without
  task-specific reprogramming.
- **Reasoning and working memory:** it has human-like common sense, critical thinking, independent problem-solving
  and continuous learning, rather than just predicting the next pattern.

How each characteristic maps onto the agent, how it is measured, and where it stands:

| characteristic | in the agent | measured by | status (v3) |
|---|---|---|---|
| Generalization | no game ids, colours or action numbers in the code; object types, roles, move vectors, floor and walls are all learned per game | score / levels on the **held-out** games (never tuned on); colour-permuted and flipped game variants | design in place; not yet measured on held-out or variants |
| Cross-domain transfer | analogy memory shared across levels and games; recipes mapped by **structure mapping** (relations over surface); skills defined on learned facts, reusable in any game | transfer ablation: held-out run with memory carried over from dev vs fresh memory (actions per level, levels) | implemented; rarely used (only 3 recipes, replay 22 actions) |
| Reasoning and working memory | common-sense priors (objects, agentness, counters, salience); working memory per level (mental map, tried actions, roles); continuous learning (memory updates every action, across games); **goal reasoning via hypothesis elimination** | share of actions spent on random fallback; levels solved; actions until the goal is identified | memory and priors in place; **goal reasoning missing**: v4 replaced random actions by systematic graph search (E8), but no goal hypotheses yet |

## 2. Setup

| item | value |
|---|---|
| Environment | ARC-AGI-3 via the `arc-agi` toolkit (`arcengine` 0.9.3), local mode, 25 public games |
| Observation / actions | 64×64 grid of 16 colours; RESET, ACTION1-5, ACTION6 (click x, y), ACTION7 (undo); available actions vary per game |
| Language / tools | Python 3.12, uv, numpy, scipy (`ndimage.label`) |
| Hardware | Apple M1, 8 GB (all runs below); bwUniCluster 3.0 A100 40 GB available for later LLM experiments |
| Split | **dev** (tuning): ar25 cd82 cn04 dc22 ft09 g50t ka59 ls20 s5i5 sk48 sp80 tn36 tr87 tu93 vc33 · **held-out** (final only): bp35 lf52 lp85 m0r0 r11l re86 sb26 sc25 su15 wa30 — `random.Random(0).sample` |
| Budget | 1000 actions per game (resets count) unless stated |
| Baseline | random agent (random available action, random pixel for clicks), same budget |

### Metric

Official toolkit scorecard (RHAE-style, `arc_agi/scorecard.py`):

- per level: `min((human_baseline_actions / agent_actions)² × 100, 115)`; 0 if not completed
- per game: level scores averaged with weight = level index, capped at 100; overall: mean over games

**Two ways of counting (from 2026-11-12):**
- **First exposure** (main number): the first play of each game; every action from the first frame counts, exploring included. All results before E15 are of this kind.
- **Best play** (second number, `run.py --clean-run`): after the first play, a *clean run* replays the shortest known solution of every level the agent solved, in a fresh play of the same game. The toolkit's scorecard takes the best play of a game. This number says how short the found solutions are, **not** how efficiently the agent learns, and is never mixed with first-exposure numbers. (Kepler, arXiv 2610.00834, reports the same kind of number with the same caveat.)

**Additional metrics (found 2026-11-12).** The benchmark's paper (arXiv 2603.24621) names four capabilities —
exploration, modeling, goal-setting, planning and execution — but reports no number per capability. These metrics
fill that gap and cover the project's AGI characteristics. `uv run python scripts/agi_metrics.py runs/<file>.json`
computes the first seven from a run file.

| # | metric | what it shows | how | v6 default today (seed 0) |
|---|---|---|---|---|
| 1 | Levels within 5× the human count | efficiency under the official rule (the leaderboard stops a level at 5× the human count) | first-exposure actions ≤ 5 × human | **2 of 7** solved levels (cd82, tn36) |
| 2 | Actions relative to the human | skill-acquisition efficiency | ours ÷ human, median over solved levels | 6.3× |
| 3 | Discovery cost | exploration: search effort per step of the solution | first-exposure actions ÷ length of the shortest route found | 28 (tn36: 1.0; sp80: 61) |
| 4 | Route vs human | planning: quality of the solution found | route length ÷ human count, median | 0.41 (shorter than the human baseline) |
| 5 | Share of actions that execute a known skill or plan | acting from knowledge vs searching (Twin: 93%) | skill usage | 7% |
| 6 | Breadth | generality across games | games with ≥ 1 level solved | 6 of 15 |
| 7 | Transfer inside a game | does knowledge carry to the next level? | ours ÷ human per level; should fall | vc33: 5.4 → 23.1 (rises: no transfer) |
| 8 | Prediction accuracy | modeling: does the agent know what an action will do? | before each action predict whether the screen will change; share right | 85% right, guess made for 98% of actions (commoner-outcome baseline 62%) — E18 |
| 9 | Goal precision | goal-setting | share of won levels where the goal in test was true on the winning screen; first hypothesis right (Twin: 87%) | 1 of 1 case, and that goal had been given up before the win — E18 |
| 10 | Generalization gap | generalization | every metric on held-out minus dev (`agi_metrics.py --vs`) | built; not run (held-out reserved) |
| 11 | Renaming robustness | generalization | same games with colours permuted and the grid mirrored (`run.py --rename N`); change in levels / score | **not robust:** score 0.28 plain vs 0.04, 0.13, 0.13 renamed; levels 7 vs 6, 7, 6 — E18 |
| 12 | Memory ablation | cross-game transfer | memory carried from earlier games vs fresh memory (`run.py --fresh-memory`) | carried 7 levels, fresh 6; same score: no systematic transfer — E18 |
| 13 | Cost | practicality | agent time per action; LLM calls | 4.4 ms per action; 0 LLM calls — E18 |

Facts from the paper that matter for reading our numbers: the human baseline is the **second-best first-run human of
ten testers** per level (not a median); the official per-level score is capped at 1.0 (the toolkit allows 1.15, which
changes none of our totals); official runs stop at **5× the human count** per level — by that rule five of our seven
first-exposure levels would not count, and the first-exposure score would be 0.26 instead of 0.28.

Secondary: levels completed, game overs. Exploration metrics (`scripts/info_gathering.py`, 300 actions, identical
observer for every agent): **useful actions** (frame changed outside the HUD), **new screens** (distinct HUD-masked
states), **wasted actions** (same state-action repeated), **click coverage** (share of the first frame's object types
acted on).

## 3. Experiments

### E1 — How many distinct things does a game show? (2026-11-04)
- **Question:** can grouping pixels into objects reduce what must be tested?
- **Method:** first frame of all 25 games; objects = same-colour connected components; type = colour + exact shape (`scripts/cluster_objects.py`).
- **Result:** 4-190 objects but only 3-17 types in 24/25 games (outlier tr87: 46). Click games: 4096 possible clicks → 7-16 types.
- **Conclusion:** test one member per type → ~250-600× fewer click probes. Exact shape is too strict for some games.

### E2 — Does clustering help exploration? (2026-11-04)
- **Method:** 7 click-only games, 150 actions: random clicks vs one-click-per-type vs one-click-per-type + reuse across levels/resets. One no-op click per level to find the HUD.
- **Result:**

  | strategy | effective clicks | types probed | levels | game overs |
  |---|---|---|---|---|
  | random | 106 | – | 0.3 | 15 |
  | cluster | 196 | 321 | 2 | 12 |
  | cluster + reuse | 192 | 255 | 2 | 13 |

- **Conclusion:** clustering ~2× effective actions and is the only strategy that cleared levels; reuse cuts probes ~20% at no cost. ft09 / lp85 unaffected by single-click probing.
- **Pitfall found:** 4/7 games have a step counter that changes every action; without masking it every click looks effective and the agent clicks the shrinking bar ("new shape = new type").

### E3 — Object map and roles (2026-11-04)
- **Method:** type + position map of ls20 and object tracking across one move (`scripts/object_map.py`).
- **Result:** one move revealed ~4 structural hypotheses: two co-moving parts = player; shrinking bottom bar = step budget; board changes only because the player stands on it; two framed patterns = likely goal comparison (untested).
- **Conclusion:** separate identity (type) from position; give types meaning (roles) from observed effects; goal hypotheses are relations between roles.

### E4 — Agent v1 (2026-11-08)
- **Agent:** perception + 5 fundamental skills (probe, replay, repeat, click_all, greedy go_to) + analogy memory (type roles, recipes, skill values); explore vs exploit by skill value.
- **Result (dev, 1000 actions):** random 0.00 / 1 level; **v1 0.24 / 2 levels** (tn36, vc33).
- **Exploration metrics (300 actions):**

  | | random | v1 |
  |---|---|---|
  | useful actions | 47% | 69% |
  | new screens | 898 | 682 |
  | wasted actions | 55% | 63% |
  | click coverage | 36% | 74% |
  | all first-frame types tried | 0/11 | 4/11 |

- **Conclusion:** good at learning what object types do in click games; poor spatial exploration in movement games (ls20: 8 screens, 93% wasted — greedy go_to bumps into walls).

### E5 — Agent v2: mental map + route planning (2026-11-08)
- **Change:** learned map (floor colours from where the player steps, walls from blocked/fatal moves, visited places) + BFS planning for `go_to` and new `explore_map`; any action that moves the player counts as a move; HUD margin scales with grid size and ignores the player's own changes.
- **Result:** new screens 682 → **1254** (> random 898); wasted 63% → **38%**; ls20 8 → 43 screens, tr87 27 → 119. Dev score 0.01 / 2 levels (sp80, vc33).
- **Conclusion:** spatial exploration fixed. Score change is one level (tn36 lost, sp80 gained); a single fast level dominates the squared metric, so levels and exploration metrics are the more reliable signal.

### E6 — Agent v3: counter fix + structure mapping (2026-11-08)
- **Change:** border bands that changed on their own are never probed/clicked; analogy by **structure mapping** (Gentner): recipe steps store an object's relational description (odd one out, multiplicity, relative size, side of the player) + surface; replay maps one-to-one onto the best-scoring object, relations weighted above surface.
- **Result:** vc33 counter clicks 13 → 0 (8 distinct types in 8 clicks). **Dev 0.09 / 3 levels** (sp80, tn36, vc33) — most levels so far. Exploration: 1242 screens, 38% wasted, 74% coverage.
- **Check:** synthetic test — level 2 changes colours and positions; replay still picks the new odd-one-out.

### E7 — Weakness diagnosis, v3 (2026-11-08)
- **Method:** per-game breakdown of where the 1000 actions go (dev games).
- **Result:** random fallback 8997/15000 actions; probe 2915; repeat 1316; explore_map 1185; go_to 202; click_all 172; replay 22.

  | # | weakness | evidence | root cause |
  |---|---|---|---|
  | 1 | nothing to do after exploring | click games: ~30-50 probes, then 900+ random | no goal hypotheses; click games need sequences / combinations |
  | 2 | ignores the step budget | 4-42 game overs per game | counter seen but not treated as time left |
  | 3 | endless novelty | sk48: 987 probes, 154 "types" | exact-shape type key; changing objects always look new |
  | 4 | movement games never test touching | ls20/tr87/g50t/tu93: 0 types tried | go_to gives up too fast; floor learned too narrowly |
  | 5 | repeat loops | ar25: repeat 754/1000 | repeat restarts on the same action |
  | 6 | analogy rarely used | replay 22 actions | only 3 recipes (3 wins) |

- **Conclusion:** the agent gathers basic information well (object effects, controls, walkable space) but cannot yet turn it into a goal — the original research question.

### E8 — Agent v4: graph exploration (2026-11-08)
- **Question:** does the best published non-LLM technique (graph exploration, arXiv 2512.24156) fix weakness 1 (random fallback)?
- **Change:** every HUD-masked screen is a node, every action tried there an edge (clicks keyed per object, so identical buttons stay distinct); actions carry a salience tier by object size; when no skill applies, do an untried action of the lowest tier here, else walk the shortest known path to the nearest screen with one, raising the tier only when nothing is reachable. Game-over actions count as tried (avoids the paper's reset-loop bug).
- **Check:** synthetic lights puzzle (3 identical buttons, win only at on-off-on) solved by graph exploration, never random.
- **Result (dev, 1000 actions):** **7 levels** (cd82, ls20, sp80, tn36, tu93, vc33 ×2), score 0.15. Random fallback 8290 → **95** actions; frontier 11664; replay 708 (7 recipes). Exploration (300 actions): 1287 screens, wasted 38% → **26%**, click coverage 74% → 64%.
- **Conclusion:** systematic graph search instead of random actions more than doubles levels (3 → 7) and opens movement games (ls20, tu93 first levels). Replay now fires often but has low value (0.06 per step): recipes are stored but frequently fail on the next level. Coverage dropped because probing hands over to the frontier earlier.

### E9 — Reproducibility fix and 3-seed baseline (2026-11-08)
- **Problem found:** screen ids used Python's `hash()`, which is randomised per process; two runs of the same v4 code gave 7 and 6 levels. Single-run comparisons E4-E8 therefore carry hidden run-to-run noise.
- **Fix:** stable screen id (`sid()`, blake2b of the HUD-masked grid); `run.py` and `scripts/info_gathering.py` re-execute with `PYTHONHASHSEED=0`. Two identical runs now give identical results.
- **Method:** `uv run run.py dev --seeds 3` (seed = game seed + agent seed, fresh memory per seed), v4 vs random.
- **Result:**

  | agent | levels per seed | levels mean ± sd | score per seed | score mean ± sd |
  |---|---|---|---|---|
  | random | 1, 2, 1 | 1.33 ± 0.58 | 0.00, 0.32, 0.01 | 0.11 ± 0.18 |
  | **v4** | 7, 7, 6 | **6.67 ± 0.58** | 0.15, 0.15, 0.15 | **0.15 ± 0.00** |

  v4 levels by game (all seeds): cd82 1, ls20 1, sp80 1, tn36 1, vc33 2; tu93 1 in 2 of 3 seeds.
- **Conclusion:** v4 beats random by ~5 levels, far outside the seed spread. The score is fragile: random's seed 1 completed sp80 quickly once (4.76) and nearly matched v4's mean score, so **levels are the primary comparison number** and the score is reported alongside. The games are almost deterministic under the seed (only tu93 varied).

### E10 — Which returned frame shows the won / lost level? (2026-11-12)
- **Question:** does the agent already receive the winning screen (idea I24)? Reading the engine (`arcengine/base_game.py`) suggested yes.
- **Method:** `uv run python scripts/win_frame.py` — v4 on cd82, ls20, sp80, tn36, vc33, 600 actions each; at every level-up and game over: number of frames the action returned and each frame's difference to the screen before the action.
- **Result:**
  - 6 level-ups: each returned ≥ 2 frames (2, 2, 2, 10, 16, 20). The **last frame is the next level** (10–71% of cells differ); the **frame before it still shows the old level in its winning state** (1–17% differ).
  - 41 game overs: the final screen differs 0–4% from the one before. They come at fixed intervals per level — cd82 ~100 actions, ls20 132 (level 2: 66–67), sp80 31 (level 2: 40–46), tn36 62, vc33 ~51 — i.e. they are consistent with **step-limit timeouts**, not deadly objects (one irregular interval in sp80).
  - Several games return many animation frames per action (cd82 15, sp80 22–28, tn36 6, ls20 6); the agent reads only the last one.
- **Conclusion:** (1) the win screen is available as `frame[-2]` at a level-up — a free positive example of the goal. (2) The agent's assumption that `frame[-1]` at a level-up is the new level holds in all cases seen. (3) Every attempt has a fixed budget of about 31–132 actions and a reset restores the level, so **a level must be solved inside one attempt**; game overs are a clock, not mistakes (random and v4 both show ~180 game overs per 15,000 actions, run files below). (4) Where the agent marked an object type "deadly", it was almost certainly a timeout (logbook F16).

### Run files for the baseline (2026-11-12)
`runs/2026-11-12-v4-baseline.json`, `runs/2026-11-12-random-baseline.json` (commit `c73cb40`, 3 seeds): identical to E9 — v4 7, 7, 6 levels (6.67 ± 0.58), random 1, 2, 1 (1.33 ± 0.58). New numbers from the records: median actions per completed level v4 324 (n = 20), random 279 (n = 4).

### E11 — A goal from the winning screen, measurement only (2026-11-12)
- **Question:** does contrasting the winning screen with every non-winning screen of the level give a goal (I24), and does the goal learned on level k pick out the winning screen of level k+1?
- **Method:** generic situation features (per colour and size class: number of objects, cells, distance to the controlled thing; totals of objects, shapes, colours, colours occurring once). Goal schema = features whose value on the win screen was never seen before in the level (weight 2) or differs from the level's start (weight 1). **Goal rank** = share of non-winning screens of level k+1 that the level-k goal scores at least as high as the real winning screen (0 = best, 0.5 = chance). No behaviour change. Run file `runs/2026-11-12-v5-step2-goal-measure.json` (commit `2603e0f`).
- **Result:** behaviour identical to v4 (7, 7, 6 levels). A schema was learned at 20 of 20 level-ups. Goal rank could only be measured where a second level was won — vc33 level 2, the same case in all 3 seeds: **0.23** (n = 3). Learned goals (seed 0, strongest features): ls20 "count colour 9 down, distance to colour 9 up, cells colour 3 up" (never seen before); sp80 "count colour 11 to 0" (never seen before); tn36 "types down, objects down, count colour 11 down"; vc33 "cells colour 0 up"; cd82 only weak features (nothing never-seen).
- **Toy check:** "collect" world (click coins until none is left): learned "count of the coin colour to 0 (never seen before)", distractors not in the goal, goal rank 0.0.
- **Conclusion:** the contrast works and finds the right goal where the goal is expressible (toy world). On the real games the evidence is one case, and most learned goals are coarse (cell and object counts).

### E12 — Using the goal; deadline rule; short replay (2026-11-12)
- **Change (commit `b4c08d6`):** (a) *goal*: the goal learned at a win is carried to the next level and pursued first — `go_to` walks to the goal's colours; the graph search is best-first (progress toward the goal minus walking cost) and orders actions by their known effect (operator table = means-ends analysis); given up when progress stalls for half an attempt. (b) *deadline*: a game over at the step limit is a timeout — no "deadly" role, no wall, no skill penalty. (c) *replay5*: replay only recipes of at most 5 actions. Each part can be switched off (`--off`).
- **Result (dev, 3 seeds, 1,000 actions):**

  | variant | run file (`runs/2026-11-12-…`) | levels per seed | levels mean ± sd | score | median actions per completed level |
  |---|---|---|---|---|---|
  | v4 baseline | `v4-baseline` | 7, 7, 6 | 6.67 ± 0.58 | 0.15 | 324 (n = 20) |
  | all three on | `v5-step3` | 6, 6, 6 | 6.00 ± 0.00 | 0.13 | 218.5 (n = 18) |
  | deadline + replay5 (goal off) | `v5-step3-off-goal` | 6, 6, 6 | 6.00 ± 0.00 | 0.13 | 218.5 (n = 18) |
  | all three off (control) | `v5-all-off` | 7, 7, 6 | 6.67 ± 0.58 | 0.15 | 324 (n = 20) |
  | only goal | `v5-only-goal` | 7, 7, 6 | 6.67 ± 0.58 | 0.15 | 324 (n = 20) |
  | only deadline | `v5-only-deadline` | 6, 6, 6 | 6.00 ± 0.00 | 0.15 | 277 (n = 18) |
  | only replay5 | `v5-only-replay5` | 6, 6, 6 | 6.00 ± 0.00 | 0.13 | 218.5 (n = 18) |

  With all three off the new code reproduces v4 exactly (same levels, same skill usage). Per game: the deadline rule alone loses sp80's first level (3 of 3 seeds) and wins tu93's in 3 of 3 (v4: 2 of 3); short replay alone loses tu93's level and changes speeds (sp80 score 0.07 → 0.16, tn36 1.81 → 1.35). The seeds are almost identical runs (games and agent are nearly deterministic), so each of these is **one** observation of a fragile level, not three.
- **Diagnosis on second levels** (cd82, ls20, sp80, tn36, vc33, seed 0): the carried goal is pursued for 30–141 actions, then given up. Best progress reached: cd82 0.83, ls20 0.18, sp80 0.00, tn36 0.33, vc33 level 2 **1.00 without the level ending**, level 3 0.55.
- **Toy worlds** (`test_agent.py`): "reach" world, level 2 with nearer distractors: 16 actions with the goal (the shortest route) vs 18 without; "collect" world: 5 vs 5 (plain exploring is already minimal there); colours renamed and grid mirrored: identical action counts.
- **Conclusion:** **negative result on the dev games.** Using the learned goal changes neither levels nor score: the goals learned from real wins are too coarse to guide (vc33: fully "reached" without a win) or cannot be advanced inside one attempt (sp80). The mechanism itself works where the goal is expressible (toy worlds). This is the limit foreseen in the plan: the feature language cannot say what these games want — the reason for the LLM reasoner (E13) and for richer features. **Decision:** by the rule "kept only if it helps", the default agent is v4 + goal reasoning (`v5-only-goal`, 6.67 ± 0.58 levels); the deadline rule and short replay stay available as switches but are off by default, although E10 shows the deadline rule is factually right — its effect on levels is one fragile level each way.

### E13 — LLM goal proposer: mechanics only, no real model yet (2026-11-12)
- **Change (commit `e60cd17`):** `llm.py` — one call to any OpenAI-compatible endpoint (unset or failing → the core plays alone), the screen as an image, one fixed prompt for every game built from the agent's measured facts. The LLM answers with goals as **conditions over the generic features** (data, not code). The core drops proposals outside the language, rejects those already true on a non-winning screen, pursues the rest in order, and refutes one in play when it becomes true without the level ending. Asked when the agent has looked around (12 actions) and has no goal to pursue, at most twice per level.
- **Check with a stand-in LLM** (`test_agent.py`): 4 proposals on a fresh "reach" level — one not in the language (dropped), one true from the start (rejected), one already true on a screen visited earlier (rejected), one right (pursued): level won in 26 actions with the LLM goal true on the winning screen; the core alone did not finish this fresh level in 300 actions. The HTTP call is tested against a local stand-in server (request shape, image, failure → no LLM).
- **Not done:** no run with a real model. Scripts for bwUniCluster (`cluster/setup.sh`, `cluster/eval.sbatch`; model `cyankiwi/Qwen3.6-27B-AWQ-INT4`, about 20 GB, 4-bit, vision) are written but untested: there was no open cluster session.

### E14 — Score accounting: where the points are (2026-11-12)
- **Question (reverse engineering of the target 0.9):** where does the score come from, and what is the cheapest way to 0.9?
- **Method:** the scorecard's own per-level actions, level scores and human action counts are saved in the run records; `scripts/score_anatomy.py <run file>` lists the points per level. One point = one unit of a game's 0–100 score; 0.9 overall = 13.5 points over 15 games.
- **Result (v5 default, seed 0):**

  | game, level | human actions | ours | points now | points at human speed |
  |---|---|---|---|---|
  | cd82 L1 | 55 | 223 | 0.29 | 4.76 |
  | ls20 L1 | 22 | 330 | 0.02 | 3.57 |
  | sp80 L1 | 39 | 324 | 0.07 | 4.76 |
  | tn36 L1 | 32 | 45 | 1.81 | 3.57 |
  | tu93 L1 | 19 | 407 | 0.00 | 2.22 |
  | vc33 L1 / L2 | 7 / 18 | 38 / 517 | 0.12 / 0.01 | 3.57 / 7.14 |
  | sum | | | 2.32 → score 0.15 | 29.60 → score 1.97 |

- **Conclusion:** the score is lost in speed, not in levels. The seven levels already solved are worth 1.97 at human speed; 0.9 is 46% of that. The winning attempts themselves are short; the waste is everything before them. So 0.9 needs no new level — only the found solutions without the waste (E15), or found sooner (E16, E17).

### E15 — Clean run: the best-play number (2026-11-12)
- **Change (commit `ef8464b`):** at each win the agent stores `short` (shortest known route through its screen graph to the winning action, with the screen expected after each step) and `actual` (the attempt that won). `run.py --clean-run` starts a fresh play and replays `short` level by level, checking every step; on a mismatch it resets the level once and replays `actual`. No game-specific code; relies on the games being deterministic.
- **Result (dev, 3 seeds; run file `runs/2026-11-12-v6-clean-run.json`):**

  | counting | levels per seed | score per seed | score mean ± sd |
  |---|---|---|---|
  | first exposure | 7, 7, 6 | 0.15, 0.15, 0.15 | 0.15 ± 0.00 (unchanged) |
  | **best play** | 7, 7, 6 redone in the clean run | 1.88, 1.91, 1.77 | **1.85 ± 0.07** |

  The clean run needs 79, 77 and 57 extra actions per seed for all 15 games. Replay length vs human count (seed 0): cd82 7 vs 55, sp80 6 vs 39, tn36 7 vs 32, vc33 5 vs 7 and 7 vs 18, ls20 25 vs 22, tu93 22 vs 19.
- **Conclusion:** the target 0.9 is reached in the best-play number (1.85). The routes the blind search finds are, in five of seven levels, much shorter than the human action counts; the agent's problem is the cost of *finding* them. First exposure is unchanged, as it must be.
- **Caveat:** best play is not a measure of learning. It is only possible where the same game can be played twice.

### E16 — Reset as a jump back to the level's start (2026-11-12)
- **Idea (I43):** attempts are short and a reset restores the level (E10), so a reset is a 1-action jump the graph search could use on purpose instead of walking on until the timeout.
- **Version 1 (commit `41464c8`): the jump is always on offer.** Run files `v6-reset`, `v6-cluster-reset`: first exposure **4, 4, 4 levels, score 0.04** (default: 6.67 / 0.15); best play 1.35. ls20, tn36 and tu93 lost their level.
- **Cause (traces, 400 actions):** ls20 — after the third attempt the agent reset after every single action (91 resets): the remembered id of the level's start screen was stale (the counter mask had grown), so the route search aimed at screens that no longer exist. tn36 — 40 resets, 358 of 360 clicks on small objects: the search kept jumping back to untried small objects elsewhere instead of trying larger objects where it stood (the winning click is on a medium-sized object).
- **Version 2 (commit `cc09b6a`):** the start id is refreshed at every fresh attempt; the jump is offered only at a dead end (nothing untried reachable by going on). Run file `v6-reset-deadend`: **identical to the default in every number** — on the dev games a dead end with untried actions elsewhere never occurs.
- **Toy world** (one-way switches, step limit 25): 33 actions with the jump, 55 without.
- **Conclusion:** negative result. Version 1 hurts, version 2 does nothing on the dev games. The switch stays off. The stale start id was a real bug whose fix helped on its own (see E17).

### E17 — One test per cluster inside the graph search (2026-11-12)
- **Observation (traces of first levels):** tn36 level 1 took 52 actions although its shortest route is 8: after the useful clicks, the graph search clicked 39 identical tiles one after another — each object is a separate untried action — while the winning click on another kind of object waited. vc33 level 1: 44 actions, route 5.
- **Change:** the project's first idea (I5: test one member per cluster) applied inside the graph search — when one member of an object type has been clicked on a screen and the type has never done anything, its twins on that screen wait until everything else has been tried. A type that ever had an effect keeps all members in play. Plus the start-id refresh from E16.
- **Result (dev, 3 seeds):**

  | variant | run file (`runs/2026-11-12-…`) | first exposure: levels | first exposure: score | best play: score |
  |---|---|---|---|---|
  | v5 core + clean run | `v6-clean-run` | 7, 7, 6 | 0.15 ± 0.00 | 1.85 ± 0.07 |
  | + cluster rule | `v6-cluster` | 7, 7, 6 | 0.27 ± 0.00 | 1.85 ± 0.07 |
  | + cluster rule + start-id refresh (**new default**) | `v6-default` | **7, 7, 7** | **0.28 ± 0.00** | **1.97 ± 0.00** |

  tn36 level 1: 52 → 13 actions (human 32) — full points for that level. tu93's level is now won in all three seeds. vc33 level 1 unchanged (38 actions). With the refreshed start id every clean-run route is at or below the human count, so best play equals the value of the solved levels at human speed (1.97). Exploration metrics (300 actions): useful actions 61%, new screens 1,294, wasted 23%, click coverage 74% — slightly better than before.
- **Conclusion:** kept and on by default. First exposure 0.15 → 0.28 comes almost entirely from one level (tn36); it is one observation, not three (seeds repeat the same run).

### Where the target stands (2026-11-12)
- **Best play: 1.97** — the target of 0.9 is reached, with solutions the agent found itself; it is not a measure of learning efficiency.
- **First exposure: 0.28 in the plain view, but only 0.04–0.13 when the games are shown with renamed colours and mirrored (E18); a fair figure is about 0.15.** By the accounting of E14, 0.9 would need two more first levels found within the human action count: cd82 (55 actions; the agent needs 223, its route is 5) and sp80 (39; the agent needs 245, its route is 4). Their solutions are short; what is missing is knowing what to aim for before the first win — the job of the LLM reasoner (E13, not yet run with a real model).

### E18 — New measurements: prediction, goal precision, renaming, memory, cost (2026-11-12)
- **Built (commit `ee86885`):** the agent predicts before every action whether the screen will change (from the same action on the same screen, else the object type's history, else the action's history) and scores itself; it records whether the goal in test was true on the winning screen; `run.py --rename N` shows every game with colours permuted and the grid mirrored (clicks mapped back); `--fresh-memory` empties the memory before every game; agent time per action is recorded; `scripts/agi_metrics.py --vs` compares runs.
- **Runs (dev, seed 0 unless stated):** `v6-metrics` (plain, 3 seeds), `v6-rename1/2/3`, `v6-fresh-memory`.

  | | plain | renamed 1 | renamed 2 | renamed 3 | fresh memory |
  |---|---|---|---|---|---|
  | levels (first exposure) | 7 | 6 | 7 | 6 | 6 |
  | score (first exposure) | 0.28 | 0.04 | 0.13 | 0.13 | 0.28 |
  | score (best play) | 1.97 | 1.74 | 1.92 | 1.74 | 1.83 |
  | solved levels within 5× the human count | 2 | 1 | 3 | 2 | 2 |
  | discovery cost (actions per route step) | 28 | 36 | 44 | 32 | 36 |
  | predictions right (baseline) | 85% (62%) | 87% (59%) | 87% (61%) | 88% (60%) | 86% (62%) |
  | agent time per action | 5.1 ms (4.4 alone) | 5.6 | 5.6 | 5.6 | 5.7 |

- **Prediction:** a guess is made for 98% of actions and is right 85% of the time (always guessing the commoner outcome: 62%). By basis (seed 0): same action seen on the same screen 99%, object type's history 85%, action's history 80%. The same action on the same screen gave the same screen 96% of about 7,000 times. The prediction is only "does the screen change", not what changes.
- **Goal precision:** one case (vc33 level 2): the carried goal was true on the winning screen, but the agent had given it up before the win. Too few cases for a rate.
- **Renaming robustness — the main finding:** the result depends on what the game looks like. tn36's first level (worth 3.57 of the 4.1 first-exposure points) is won in the plain view and **lost in all three renamed views**; ls20's first level takes 368 actions in the plain view and 35 in two of the renamed views; s5i5's is won in one renamed view only. Over the four views the first-exposure score is 0.04–0.28 (mean 0.15) and levels are 6–7. Best play stays above the target in every view (1.74–1.97).
- **Cause in tn36 (traces):** with memory from eleven earlier games the agent goes straight to the graph search. Its order of untried objects inside one size tier follows the colour numbers. In the plain view the six small switches come first (won after 13 actions); renamed, the 40 tiles come first, two of them change something, so the cluster rule keeps all tiles in play, and the level is not won in 120 actions.
- **Memory ablation:** with an empty memory per game: 6 levels (tu93 lost), same score, tn36 in 20 instead of 13 actions, sp80 in 214 instead of 245. No systematic gain from carrying memory across games; what is carried mostly changes which skill runs.
- **Generalization gap:** the comparison is built (`agi_metrics.py dev.json --vs heldout.json`); not run — the held-out games stay untouched until the user decides.
- **Conclusion:** the first-exposure gain of E17 (0.15 → 0.28) is **not robust**: it came from an accident of object order. A fair first-exposure figure today is about 0.15 (0.04–0.28 over four views). Renamed views are the better "seeds" for this agent (F31). The measurements themselves work and stay.

## 4. Summary of results

| version | dev score | dev levels | new screens | wasted | click coverage |
|---|---|---|---|---|---|
| random | 0.00 | 1 | 898 | 55% | 36% |
| v1 | 0.24 | 2 | 682 | 63% | 74% |
| v2 | 0.01 | 2 | 1254 | 38% | 75% |
| v3 | 0.09 | 3 | 1242 | 38% | 74% |
| v4 | 0.15 | 7 | 1287 | 26% | 64% |
| v4, 3 seeds | 0.15 ± 0.00 | 6.67 ± 0.58 | | | |
| random, 3 seeds | 0.11 ± 0.18 | 1.33 ± 0.58 | | | |
| v5 core (v4 + goal reasoning), 3 seeds | 0.15 ± 0.00 | 6.67 ± 0.58 | | | |
| v5 with deadline rule + short replay, 3 seeds | 0.13 ± 0.00 | 6.00 ± 0.00 | | | |
| v5 core + clean run — **best play** (not first exposure), 3 seeds | 1.85 ± 0.07 | 6.67 ± 0.58 | | | |
| **v6 default** (cluster rule in the graph search), first exposure, 3 seeds | 0.28 ± 0.00 | 7.00 ± 0.00 | 1294 | 23% | 74% |
| v6 default — **best play** (not first exposure), 3 seeds | 1.97 ± 0.00 | 7.00 ± 0.00 | | | |

Rows without ± are single runs made before the reproducibility fix (E9) and carry run-to-run noise of about one level. Held-out split not yet evaluated.

## 5. Design decisions

| decision | by | reason |
|---|---|---|
| Develop on 15 dev games, report final numbers on 10 held-out games | joint | instructor tests unseen environments; public set falls to trivial tricks (AERA, arXiv 2605.25931) |
| Official scorecard as primary metric + exploration metrics | joint | comparable to the benchmark; exploration metrics are more stable at low level counts |
| Known object type at level start: one cheap confirmation; new type: full information gathering | user | reuse knowledge cheaply, explore only what is new |
| Never probe a type that was deadly before | Claude (accepted) | a confirmation probe could cost the attempt |
| Skills only use learned facts (types, roles, move vectors, floor) — no game ids, colours or action numbers | user | generalisation to unseen environments |
| Movement: build a mental map first, then plan the route | user | greedy movement got stuck (E4) |
| Analogy between levels / games by structure mapping (relations over surface) | user | colours and positions change between levels |
| LLM adviser later, optional; core runs without it | user | instructor's machine may have no GPU / network; LLM tunes settings, not every action |
| Everything is logged for the report: process, every idea, every failure, a run file behind each number | user (2026-11-12) | the report must be written from the record |
| The reasoning must be general: one loop for every game and question, no hand-written goal types | user (2026-11-12) | unseen environments |
| Hybrid: a local LLM on bwUniCluster reasons, the core checks and plays alone without a GPU | user (2026-11-12) | best published solutions get their reasoning from an LLM (research notes §14) |
| LLM answers are data (conditions over generic features), never executed code, and always checked against the evidence | Claude (accepted with the plan) | safety; the check is what the core contributes |
| A new part becomes default only if it helps on the dev games; otherwise it stays a documented switch | joint (plan of 2026-11-12) | E12: deadline rule and short replay off by default, goal reasoning on; E16: reset jump off; E17: cluster rule on |
| Target: official score about 0.9 on the 0–100 scale, only the bwUniCluster open model as LLM | user (2026-11-12) | reverse-engineered in E14 |
| Report two numbers, never mixed: first exposure (main) and best play (with a clean run) | user (2026-11-12) | the toolkit scores the best play; the report must say which number is which |

## 6. Bugs found and fixed

| bug | found by | fix |
|---|---|---|
| Counter bar counts as an effect / gets clicked | E2 spike, E6 trace | HUD = border bands changing on 2 separate steps; suspected bands never probed |
| HUD detection masked a button near the border | self-check (`test_agent.py`) | changes on the clicked object don't vote; repeat evidence required |
| Player's own movement voted as HUD on small grids | maze self-check | player boxes excluded; margin scales with grid size |
| Greedy go_to oscillated against walls | E4 trace (ls20) | mental map + BFS route planning |
| Stale id of the level's start screen after the counter mask grew (route searches aimed at dead screens) | E16 trace: reset after every action in ls20 | id refreshed at every fresh attempt |
| Runs not reproducible (`hash()` randomised per process) | two runs of v4 gave 7 vs 6 levels | stable `sid()` screen ids + `PYTHONHASHSEED=0` in run scripts (E9) |

## 7. Task breakdown

Each task names the AGI characteristic it serves (G = generalization, T = cross-domain transfer, R = reasoning and
working memory) and a checkable done-criterion. Order = suggested sequence; later tasks build on earlier ones.

### 7a. Research tasks

| id | question | serves | method | output |
|---|---|---|---|---|
| R1 | Which kinds of goals do ARC-AGI-3 games have? | R | play each **dev** game by hand (`play.py`), write down the goal in one sentence (dev only, to avoid overfitting) | goal-template catalogue (touch, collect all, match pattern, click order, …) with frequencies |
| R2 | How to choose the action that eliminates the most goal hypotheses? | R | literature: active learning / version spaces, Tycho, OPINE-World, Kepler | selection rule for the hypothesis space (T2) |
| R3 | How to keep object identity when objects change shape, grow or rotate? | G | object tracking / persistence literature; inspect sk48, ka59, cd82 | type-key design for T3 |
| R4 | How to measure transfer and generalization honestly? | G, T | design: memory ablation (carried vs fresh), colour permutation / flip wrapper, seeds and confidence intervals | evaluation protocol for T1, T6, T7 |
| R5 | What should the optional LLM adviser adjust, and with which model? | R | test small vision models on bwUniCluster A100 40 GB (4-bit 27-31B, 7-12B); prompt = object list + image | adviser interface and model choice for T9 |

### 7b. Implementation tasks

| id | task | serves | done when | depends on |
|---|---|---|---|---|
| T1 | ✅ Multi-seed evaluation (≥ 3 seeds), levels as primary number, mean ± spread | G | `run.py --seeds N` reports per-seed and mean levels / score (done, E9) | – |
| T2 | **Goal-hypothesis space**: hypotheses = goal template × object roles; eliminated by wins, losses, failed attempts; explore = most-eliminating action, exploit = best surviving hypothesis | R | random fallback < 30% of actions; dev levels ≥ v3 | R1, R2, T1 |
| T3 | Looser type key (colour + size class + tracking) | G | sk48 / ka59 probes stop growing without bound; coverage ≥ v3 | R3 |
| T4 | go_to tries every reachable target; floor learned from the whole path | R | movement games: object types touched > 0 in ls20, tr87, g50t, tu93 | – |
| T5 | Budget-aware switch: as the step counter runs low, stop exploring and test the best hypothesis | R | fewer game overs per game than v3 | T2 |
| T6 | Transfer ablation: held-out with memory carried from dev vs fresh memory | T | table of actions per level / levels, both conditions | T1, T2 |
| T7 | Robustness variants: colour-permuted and flipped games | G | score / levels within the seed spread of the original games | T1 |
| T8 | **Held-out evaluation, once**, with the final agent | G | final results table in the report | T1-T7 |
| T9 | Optional LLM adviser on bwUniCluster (settings only, core runs without it) | R | core vs core + adviser on dev, same seeds | R5, T2 |

## 8. Reproduce

```bash
uv sync
uv run python test_agent.py                  # self-checks
uv run run.py dev --seeds 3 [--agent random]  # E9 baseline (E4-E8 used single runs)
uv run python scripts/info_gathering.py      # exploration metrics
uv run --with scipy python scripts/cluster_objects.py   # E1
uv run python scripts/win_frame.py           # E10
uv run run.py dev --seeds 3 --clean-run --label NAME   # E15: first exposure and best play
uv run python scripts/score_anatomy.py runs/<file>.json   # E14
uv run python scripts/agi_metrics.py runs/<file>.json     # additional metrics (section 2)
```
