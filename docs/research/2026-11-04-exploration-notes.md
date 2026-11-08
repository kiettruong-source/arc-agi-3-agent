# ARC-AGI-3 exploration — research notes (2026-11-04)

Working notes for a later report. Focus: exploration that discovers the **goal / end condition** of an unknown game, cheaply.

## 1. Setup

- Toolkit: `arc-agi` (PyPI) + `arcengine` 0.9.3, uv project, Python 3.12.
- `uv run play.py <game>` — play by hand in the terminal; `uv run play.py list` — 25 public games.
- Browser player: https://arcprize.org/play?task=ls20
- Frame fields per step: `state` (NOT_FINISHED / GAME_OVER / WIN), `levels_completed`, `win_levels`, `available_actions`, `frame` (64×64 grids, possibly several animation frames).
- Actions: RESET, ACTION1–4 (up/down/left/right), ACTION5 (interact), ACTION6 (click x,y), ACTION7 (undo).
- Scoring is action-efficiency relative to humans (RHAE), so every action — including resets — costs.

## 2. Landscape (as of Nov 2026)

| Family | Example | Result | Exploration style |
|---|---|---|---|
| Executable world model + verification | Tycho, Retrodict, Kepler, baseline1 | ~99–100% (public set, self-reported) | Agent writes Python simulator, verifies vs recorded frames, plans through it |
| Lightweight coding REPL | Tufa Labs "The Duck" (Qwen 3.6 27B) | 1st, 2026 Milestone #1 | Python REPL, no hand-made tools, message eviction for long play |
| Vision LLM + reflection | Reki, forge (Gemma-4-31B) | 2nd/3rd Milestone #1 | Frames as images → JSON plan of 1–4 actions; reflection every ~10 steps |
| Graph exploration (Go-Explore-like) | arXiv 2512.24156 | 3rd private, 2025 preview; median 30/52 levels, no LLM | State graph, go to nearest untested (state, action) pair, visual salience |
| CNN + RL brute force | StochasticGoose (Tufa) | 1st 2025 preview, 12.58% | Predict frame-changing actions; collapsed under efficiency scoring |

Caveats: community scores are self-reported on the 25 public games (now effectively saturated). Held-out Kaggle leaderboard top scores reported ~50–69, snapshots disagree. arXiv 2607.15439: model strength / reasoning effort matters more than harness variant; plain-text agent on a top model also cleared all public levels.

## 3. Tycho in detail (arXiv 2607.28287, github.com/NIMI-research/Tycho, Apache-2.0)

- Each level = rendered deterministic Moore machine: hidden state → (grid, available actions, outcome ∈ {ongoing, level_complete, game_over}). Not Markov in the grid alone (hidden counters, offsets).
- World model = agent-written Python: `init_state`, `transition`, `render`, `outcome`.
- Roles: **actor** (only one who acts) and optional **builder** (writes/repairs the model). Best policy: actor calls builder on demand (88.49 RHAE, Opus 4.8; 100 with GPT-5.6 Sol / Opus 5).
- Verification: replay recorded attempts through the model; every predicted cell and every recorded win/loss outcome must match. Mismatch = falsification.
- Planning: search the model for an action sequence ending in `level_complete`; take one step, re-check, replan.
- No dedicated exploration algorithm — actor notes rival hypotheses and spends actions to separate them.
- **Key gap: the goal.** Goal = just the `outcome()` function. Terminal-outcome recall only ~20–21% for best-scoring policies (77.7% under automatic builder trigger, which scored lower overall). Paper: a model can "capture mechanics while missing the objective."

## 4. Go-Explore (arXiv 1901.10995, 2004.12919)

- Loop: archive distinct states ("cells", e.g. downscaled frames) → return to a promising one (replay or restore) → explore from there.
- Strength: reliably reaches rare states, including wins and losses.
- Weakness: goal-agnostic and action-hungry — penalised by efficiency scoring.
- Role in a goal-discovery design: a cheap **generator of outcome events** (wins/losses) to feed goal hypotheses.

## 5. Cost-efficient information gathering (ranked)

1. **Zero-action info:** segment first frame into objects, detect HUD/counters, read `available_actions` (tells movement vs click vs undo game).
2. **Never pay twice:** deterministic games → each (state, action) once; group actions by effect; test one object per type ("dead-signature": stop clicking a type that did nothing).
3. **Disagreement-driven probes:** hold 2–3 rival hypotheses, take the action where they predict different frames. Actions all hypotheses agree on teach ~nothing.
4. **Reversible probes:** prefer undo (ACTION7) / short resets so risky probes are cheap.
5. **Graph frontier (Go-Explore) as fallback** when no hypotheses are left.

For goals specifically: rival goal hypotheses ("block onto square" vs "match the corner panel"); probe toward states where they disagree about whether the level ends. Signal moments (counter change, flash, level end) are the information-dense ones.

## 6. Clustering idea (human-like perception)

Humans see a few objects, not 4096 pixels. Cluster, test one representative per cluster, split a cluster when a member behaves differently (that split is itself a clue).

Measured on the first frame of all 25 games (`uv run --with scipy python scripts/cluster_objects.py`; objects = same-colour connected components, type = colour + exact shape):

| game | actions | objects | types |
|---|---|---|---|
| ft09 | [6] | 64 | 14 |
| ls20 | [1,2,3,4] | 18 | 15 |
| re86 | [1,2,3,4,5] | 23 | 7 |
| vc33 | [6] | 10 | 9 |
| lp85 | [6] | 37 | 13 |
| sc25 | [1,2,3,4,6] | 22 | 12 |
| sk48 | [1,2,3,4,6,7] | 45 | 15 |
| ka59 | [1,2,3,4,6] | 13 | 9 |
| bp35 | [3,4,6,7] | 190 | 7 |
| tu93 | [1,2,3,4] | 65 | 6 |
| sb26 | [5,6,7] | 23 | 17 |
| lf52 | [1,2,3,4,6,7] | 58 | 8 |
| su15 | [6,7] | 27 | 7 |
| tn36 | [6] | 47 | 9 |
| s5i5 | [6] | 27 | 16 |
| wa30 | [1,2,3,4,5] | 11 | 7 |
| cd82 | [1,2,3,4,5,6] | 13 | 12 |
| r11l | [6] | 37 | 12 |
| ar25 | [1..7] | 11 | 7 |
| m0r0 | [1,2,3,4,5,6] | 4 | 3 |
| sp80 | [1,2,3,4,5,6] | 7 | 6 |
| cn04 | [1,2,3,4,5,6] | 7 | 4 |
| tr87 | [1,2,3,4] | 63 | 46 |
| dc22 | [1,2,3,4,6] | 35 | 16 |
| g50t | [1,2,3,4,5] | 10 | 10 |

Click games: 4096 possible clicks → 7–16 types (~250–600× fewer probes). Outlier tr87 (46 types) shows exact-shape matching is too strict.

Clustering levels: pixels → objects → types (colour+shape) → **roles** (wall / pushable / button / inert / deadly, by observed behaviour); actions → effect classes; states → abstract cells (Go-Explore). Roles make goal hypotheses short ("move the pushable onto the odd-coloured thing").

Known limits: HUD elements counted as objects; same colour ≠ same behaviour.

## 6b. From clusters to a map (identity vs position)

Map = list of (type, position). Type is position-free (colour + shape cropped to its bounding box); position is separate. Track objects across frames by matching type + nearest position -> motion.

ls20 example (`uv run --with scipy python scripts/object_map.py ls20 ACTION1`), one ACTION1 step:
- O + K moved y45->y40 together -> co-moving parts = one composite = likely the player.
- N (bottom bar) shrank -> likely a move budget / counter (game has a step limit).
- C (board) "changed" only because the player sits on it -> big regions need a background layer, compared per cell.
- E (+ I/J pattern, top) and F (+ L/M pattern, bottom-left) are static framed patterns -> hypothesis: goal = make them match (untested).
One action -> ~4 structural hypotheses.

Fixes for the naive version: background layer for large regions; identify resizing things by colour + position and read size change as a counter value; merge parts after they co-move twice.

Goal hypotheses as **relations** over the map: distance(player, X), inside(I, E), pattern(E) == pattern(F), aligned(a, b), touching(player, X). Check each against every recorded win/loss.

## 6c. Giving clusters meaning (roles from observed effects)

A type gets meaning from its observed effects (affordances), not its looks:
- moves with arrow actions -> player; blocks player -> wall; moves when pushed -> pushable
- disappears on contact (+ counter change) -> pickup; contact -> GAME_OVER -> deadly; contact -> level_complete -> goal/target
- click changes something elsewhere -> switch; shrinks/grows every action -> counter/budget
- static but mirrors something -> display/hint; no effect -> inert (stop testing)

Why it pays: (1) transfers across levels (same colour+shape ~ same role, no re-testing); (2) goal hypotheses become few, short role predicates ("pushable on target", "collect all pickups", "display matches"); (3) steers exploration (avoid deadly, skip inert, probe unknowns).

Store roles as evidence counts ("wall: seen 3, contradicted 0"), not fixed labels. A member that deviates splits the cluster - usually the game's twist (e.g. door is a wall until key collected). State-dependent roles are where an executable transition model (Tycho) must take over.

ls20 so far: O+K = player (1 move), N = counter, E/F = display (unconfirmed), C = unknown.

## 6d. Reuse rule at level start (user's decision)

1. Cluster the new frame (0 actions).
2. Known type (same colour+shape): reuse its role, confirm with ONE cheap probe; pass -> trust for the level; fail -> treat as new.
   New type: full information gathering (one member, reversible probes first, disagreement-driven).
3. New game: everything is new.

Calls made by Claude (to confirm): same colour + different shape = "probably same", quick check; deadly roles are never probe-confirmed, only avoided; roles carry across levels within a game, not across games (yet).

## 6e. Spike: random vs cluster vs cluster+memory (click games)

Setup: 7 click-only games (ft09, vc33, lp85, tn36, s5i5, r11l, su15), 150 actions each. Pure exploration, no goal pursuit.
- random: random pixel clicks (mean of 3 seeds)
- cluster: click one member per untested type; then other members of types that had an effect; skip inert types
- cluster+mem: cluster + 6d reuse rule (roles remembered across levels / resets; known-active get one cheap check, known-inert skipped)
"Effective" = click changed the frame outside the HUD. HUD found by 1 no-op background click per level (same for all strategies).

| total over 7 games | effective clicks | types probed | levels | game overs |
|---|---|---|---|---|
| random | 106 | - | 0.3 | 15 |
| cluster | 196 | 321 | 2 | 12 |
| cluster+mem | 192 | 255 | 2 | 13 |

Per game (effective clicks, random -> cluster): tn36 3.7 -> 42 (+1 level), s5i5 3.7 -> 29, vc33 0.3 -> 7, r11l 79 -> 94 (+1 level vs 0.3), su15 7.7 -> 11, ft09 / lp85 no difference.

Takeaways:
- Clustering ~2x effective actions; only clustered strategies cleared levels (tn36, r11l L1) with zero goal knowledge.
- Reuse rule: ~20% fewer type probes, same effect and levels (savings come from not re-probing after resets).
- ft09 / lp85 unaffected -> single-click probing isn't what matters there; candidate for deeper look.
- Pitfall found: 4/7 games have an on-screen step counter that changes every action. A first run without HUD masking showed 100% "effective" clicks and the cluster agent kept clicking the shrinking bar (new shape = "new type"). Effect detection must mask the HUD first.
- Game overs are mostly the step budget (~50 actions per attempt) running out.

Caveats: click games only; "effective" != progress; deterministic strategies single run; spike script not kept in repo.

## 6f. Better techniques than the spike & optimisation ideas

- **Graph-Based Exploration (arXiv 2512.24156)** = superset of our spike, no LLM, 3rd private (2025 preview). Single-colour connected components; probable status bars masked before hashing the frame -> state ID; directed state graph (state -> action -> outcome, successor, distance to frontier); 5 priority tiers by size, morphology, colour salience (status bars lowest); pick untested action <= priority p in current state, else go to nearest state with one, else raise p. Ablation: segmentation + random already solved vc33 x5, ft09 x2, lp85 x1 (matches our spike). Known bug: reset-triggering actions not marked tested -> reset loops. Limits: assumes deterministic + fully observable; blows up on deep levels (ft09 L6+, ls20 L3+).
- **Explore Before You Solve / AERA (arXiv 2605.25931)**: EXPLORE -> VERIFY -> PLAN with Qwen2.5-0.5B, RHAE 0.21 public, 0.30 private. Benchmark critique: 10/25 public games solvable in one blind step, 8 by repeating one action 50-200x, null-coordinate bug bypasses 18 games. => public-set level counts can't distinguish smart exploration from trivial heuristics; evaluate carefully.
- **Object-centric curiosity prototype (github Crispsec/arc-agi-3-object-centric-agent)**: object slots + IoU tracking + per-object meaning vectors, explore least-understood first. Result 0 levels; over-explored and died (tn36 died at step 61). => exploration needs a stopping rule / budget.

Optimisation plan for our spike (impact order): (1) state graph with masked-frame hashing, never repeat (state, action), route to frontier; (2) salience tiers instead of flat "untested type first"; (3) mark game-over/reset actions as tested + deadly; (4) roles conditioned on state, not per level; (5) better metric: actions per new fact learned, actions per level.

## 6g. Generalisation concept

Principle: generalise the *learning procedure*, not solutions (Chollet: intelligence = skill-acquisition efficiency). Game-specific knowledge is learned at test time; only the loop + priors transfer.

| | game-specific (learned while playing) | general (built in development) |
|---|---|---|
| what | rules, roles, goal of this game | how to explore / cluster / hypothesise / verify |
| ours | "red square = deadly" | cluster -> probe one per type -> reuse -> guess goal |

Core knowledge priors to build on: objectness (cohesion, persistence, contact), agentness (something responds to my input), goal-directedness, number / geometry / topology (counting, matching, inside / adjacent, symmetry). Reason in these terms, not pixels / colours.

Transfer by distance: step->step exact rules; level->level role meanings (6d); game->game only the role vocabulary + ranked goal templates (reach X, collect all, A matches B, fill / sort, avoid Y); unseen games: only loop + priors. Goal templates = prior for goal-guess-before-first-win.

Anti-overfitting: hold out ~10 of 25 public games, never tune on them (AERA: public set falls to trivial tricks); metrics = actions per new fact, speed of understanding a new game; every heuristic must be justified by a core prior, not a specific game.
Litmus test: still works with colours permuted and grid flipped/rotated? Clustering / roles / goal templates pass; "click colour 9" fails.

## 7. Direction being explored

User's own play loop: poke → hypothesise "put X there" → try → remember failure / success → carry to next level.
Candidate design: Go-Explore-style moves to reach outcome events cheaply → goal hypotheses as testable predicates over object roles → check against every recorded win/loss → choose next probe to split surviving hypotheses. Fills Tycho's `outcome()` gap.

**Agent concept (user, 2026-11-04):** "gather pure information, try and find meaning, cheap first, have some instinct / common sense."
- Instinct (built-in, overridable priors): world = objects, same type behaves alike; one thing is "me"; edge bars / numbers = counters (don't probe); small / rare / odd-one-out = interesting, big plain = background; contact matters, some contact kills; levels end for a reason drawn from a few goal templates (reach, collect all, A matches B, fill / sort).
- Learned per game: rules, roles, goal.
- Loop: (1) pure information gathering, cheapest first (0-action perception -> one probe per type, salient first, reversible first, never re-ask); (2) find meaning (effects -> roles; roles + goal templates -> few goal hypotheses checked against all wins / losses); instinct orders both, evidence overrides instinct (a contradiction is often the twist).
- Claude's guesses (unconfirmed): needs a stop-exploring switch to pursue the best goal hypothesis; instinct is a small hand-written list, not learned from public games.

Open questions: (a) goal guess before first win vs fast confirmation after a win; (b) DECIDED -> see 6d; (c) build on Tycho vs own small harness in this repo.

## 8. Compute: bwUniCluster 3.0 (checked 2026-11-08)

- Student account. `sinfo` blocked for users; use `scontrol show partition` and `sinfo_t_idle`.
- GPU partitions: gpu_a100_short (30 min, 4 GPU/node), dev_gpu_h100, gpu_h100_short, gpu_h100, gpu_h100_il (2 days, 4 GPU/node), gpu_a100_il, dev_gpu_a100_il, gpu_mi300 (AMD, avoid). Often busy -> queue batch jobs.
- Verified: `sbatch -p gpu_a100_short --gres=gpu:1 --time=00:05:00 --wrap="nvidia-smi"` -> NVIDIA A100-PCIE-40GB, CUDA 13.4 driver. H100 memory not verified.
- Modules: devel/cuda/12.8, devel/miniforge (python 3.12). Storage: 500 GB quota on /pfs/data6, 65 GB used.
- Fit on 1x A100 40GB: 27-31B vision model at 4-bit AWQ/GPTQ (~16-18 GB) comfortable; FP8 tight and not native on Ampere; 7-12B bf16 easy. Plan: develop with 7-12B (Mac Ollama or A100), final runs with 4-bit 27-31B. LLM adviser served via vLLM (OpenAI-compatible) so the agent only swaps base URL between Ollama / vLLM / Gemini.

## 9. Agent v1 — first results (2026-11-08)

Dev split (15 games), 1000 actions per game, official scorecard metric:

| agent | overall score | levels | notes |
|---|---|---|---|
| random | 0.00 | 1 (sp80) | |
| v1 | 0.24 | 2 (tn36 L1 score 3.57, vc33 L1) | memory: 470 types, 2 recipes |

Skill usage (v1): go_to 8361, random 3979, probe 1944, repeat 433, click_all 91, replay 8.
Skill value per step: replay 0.44, probe 0.41, click_all 0.41, repeat 0.27, go_to 0.14, random 0.06.

Reading: v1 >= random but both are near zero. go_to burns most of the budget at low value -> next fixes:
(1) cap go_to episodes / stop when value falls below explore; (2) state-graph frontier navigation instead of the
random fallback; (3) HUD detection needs repeat evidence (fixed after a self-check caught border buttons being masked).
Held-out split not run yet (reserved for final numbers).

## 10. Information-gathering metrics, v1 vs random (2026-11-08)

`uv run python scripts/info_gathering.py` — same observer for both agents, 300 actions, 15 dev games.

| metric | random | v1 |
|---|---|---|
| useful actions (change outside HUD) | 47% | 69% |
| click games: initial object types probed by 300 | 36% | 74% |
| click games: all initial types probed | 0/11 | 4/11 (ar25 @7, cn04 @12, ft09 @16, vc33 @101) |
| unique screens reached | 898 | 682 |
| wasted actions (repeat same state-action) | 55% | 63% |

- Strong: learning what object types do in click games (2x coverage; vc33 / tn36 useful actions 1-4% -> 33-34%).
- Weak: spatial exploration in movement games — ls20 78 -> 8 screens (93% wasted), tr87 192 -> 27, g50t / tu93 87-90% wasted: go_to bumps into walls.
- Coverage plateaus at 58-67% in several click games (guess: exploit value overtakes probe, or cross-game memory marks a type inert).

## 11. Plan: explicit hypothesis space

v1 has only implicit hypotheses (type roles, recipes). Next design:
- hypotheses = goal template x object types (touch(T), click_all(T), match(A,B), avoid(T)), generated from the frame
- status per hypothesis (untested / failed / survived N); wins, losses, failed attempts eliminate
- explore = action that eliminates most remaining hypotheses; exploit = execute top surviving hypothesis with its skill
- skills = how hypotheses are tested; analogy memory = prior ranking (what won before first)

Order: (1) fix go_to stuck + premature exploration stop; (2) frontier navigation instead of random fallback; (3) hypothesis space; (4) per-level attribution log; (5) held-out run once.

## 12. Agent v2 / v3 (2026-11-08)

v2 = mental map for movement (floor colours learned from where the player steps, walls from blocked / fatal moves,
visited places) + BFS route planning (`go_to` to untried object types, `explore_map` to the nearest unvisited spot);
any non-click action that moves the player counts as a move (no hard-coded action ids); HUD margin scales with grid
size and ignores the player's / clicked object's own changes.
v3 = v2 + counter-bar fix (border bands that changed once on their own are never probed / clicked) +
**structure mapping** for analogy (Gentner): recipe steps store an object's relational description (odd-one-out,
multiplicity, relative size, side of the player) + surface (type, colour, shape); replay maps one-to-one onto the
best-scoring object in the new level, relations weighted above surface, threshold 0.5.

| version | dev score | levels | screens (info, 300 act) | wasted | click cover |
|---|---|---|---|---|---|
| random | 0.00 | 1 | 898 | 55% | 36% |
| v1 | 0.24 | 2 (tn36, vc33) | 682 | 63% | 74% |
| v2 | 0.01 | 2 (sp80, vc33) | 1254 | 38% | 75% |
| v3 | 0.09 | 3 (sp80, tn36, vc33) | 1242 | 38% | 74% |

Notes: score differences are within one level (noise); levels and exploration metrics are the more stable signal.
Main remaining waste: random fallback ~8300 / 15000 actions once map + types are exhausted -> explicit hypothesis
space (goal guesses) is the next slice. Only 3 recipes learned so far, so structure mapping rarely fires yet.

## Sources

- ARC-AGI Community Leaderboard — https://arcprize.org/leaderboard/community
- ARC Prize 2026 Milestone #1 — https://arcprize.org/blog/arc-prize-2026-milestone-1
- ARC-AGI-3 Preview Agent Competition — https://arcprize.org/competitions/arc-agi-3-preview-agents
- Kaggle ARC Prize 2026 leaderboard — https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard
- Tycho — https://arxiv.org/abs/2607.28287 , https://github.com/NIMI-research/Tycho
- Kepler — https://arxiv.org/html/2610.00834
- Executable World Models for ARC-AGI-3 — https://arxiv.org/pdf/2605.05138
- Do Coding Agents Need Executable World Models… — https://arxiv.org/abs/2607.15439
- Graph-Based Exploration for ARC-AGI-3 — https://arxiv.org/abs/2512.24156
- Go-Explore — https://arxiv.org/pdf/1901.10995 ; First return, then explore — https://arxiv.org/pdf/2004.12919
- ARC-AGI-3 benchmark paper — https://arxiv.org/abs/2603.24621
- StochasticGoose background — https://untitledlife.github.io/summerslog/posts/2026-03-26-the-game-i-cannot-play.html
