# Experiment protocol — ARC-AGI-3 agent

Project: an agent for ARC-AGI-3 interactive games that discovers each game's rules and **goal** through cheap
exploration, generalises to unseen environments, and is tested by the course instructor on environments not known in
advance. Background research, sources and design notes: [research/2026-11-04-exploration-notes.md](research/2026-11-04-exploration-notes.md).

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

## 4. Summary of results

| version | dev score | dev levels | new screens | wasted | click coverage |
|---|---|---|---|---|---|
| random | 0.00 | 1 | 898 | 55% | 36% |
| v1 | 0.24 | 2 | 682 | 63% | 74% |
| v2 | 0.01 | 2 | 1254 | 38% | 75% |
| v3 | 0.09 | 3 | 1242 | 38% | 74% |
| v4 | 0.15 | 7 | 1287 | 26% | 64% |

Single seed per version; score differences of one level are within noise. Held-out split not yet evaluated.

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

## 6. Bugs found and fixed

| bug | found by | fix |
|---|---|---|
| Counter bar counts as an effect / gets clicked | E2 spike, E6 trace | HUD = border bands changing on 2 separate steps; suspected bands never probed |
| HUD detection masked a button near the border | self-check (`test_agent.py`) | changes on the clicked object don't vote; repeat evidence required |
| Player's own movement voted as HUD on small grids | maze self-check | player boxes excluded; margin scales with grid size |
| Greedy go_to oscillated against walls | E4 trace (ls20) | mental map + BFS route planning |

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
| T1 | Multi-seed evaluation (≥ 3 seeds), levels as primary number, mean ± spread | G | `run.py` reports per-seed and mean levels / score | – |
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
uv run run.py dev [--agent random]           # E4-E6 scores
uv run python scripts/info_gathering.py      # exploration metrics
uv run --with scipy python scripts/cluster_objects.py   # E1
```
