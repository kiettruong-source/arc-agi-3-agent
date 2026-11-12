# Project logbook

The raw history of the project, kept for the report: **what was done** (journal), **every idea** (also untried
and rejected ones) and **everything that did not work** (failures and lessons).

- Formal experiment record: [protocol.md](protocol.md) (experiments E1, E2, …).
- Background research and sources: [research/2026-11-04-exploration-notes.md](research/2026-11-04-exploration-notes.md).
- Evidence files: `runs/<date>-<label>.json`, written by `uv run run.py … --label <label>`.

**Rules.** An idea is logged when it comes up. A failure is logged before it is fixed. A step that does not help is
still logged and committed as a negative result. No number enters the protocol or the report without a run file
(from 2026-11-12 on; earlier numbers exist only in the protocol, see F17).

"Who": **user** = project author, **Claude** = AI coding assistant, **joint** = decided together, **paper** = from literature.

---

## 1. Process journal

### 2026-11-04 — Setup and exploration research
1. Set up the environment: uv project, Python 3.12 (system Python 3.14 was too new for some packages — Claude), `arc-agi` toolkit, `play.py`.
2. `play.py` turned into a keyboard-driven player to sanity-check games by hand; the same game opened in the browser player on arcprize.org (the local toolkit has no browser view).
3. Surveyed current ARC-AGI-3 approaches (executable world models, Tufa Labs "Duck", vision-LLM policies, graph exploration, StochasticGoose).
4. **Direction set by the user:** work on the exploration that discovers the goal / end condition of a game. The user's own play loop: click around, see what it does, guess "put this there", remember failures and successes.
5. Studied Tycho (goal = one `outcome()` function, weak) and Go-Explore.
6. User's idea: humans see clusters → cluster objects to explore cheaply. Measured on all 25 games (E1).
7. Follow-up ideas from the user: see clusters as a map although positions differ (E3); clusters get meaning after exploration; reuse rule for known vs new clusters.
8. Quick test random vs cluster vs cluster + reuse on 7 click games (E2). Found the step-counter pitfall (F2).
9. Checked for better published techniques: graph-based exploration, AERA's critique of the public games, an object-centric curiosity prototype that scored 0 levels.
10. Discussed generalisation; the user's agent concept: "gather pure information, try and find meaning, cheap first, have some instinct / common sense".
11. Research notes file started; nothing committed yet.

### 2026-11-08 — Goal of the project, compute check, agent v1 → v4
1. **Goal stated by the user:** a report plus an implementation that the professor tests on various, unknown environments; the aim is an agent with AGI characteristics.
2. User's wishes: work on fundamentals; a cheap LLM adjusts the agent's settings; the LLM should also see the screen. User has bwUniCluster access.
3. Compute check: bwUniCluster 3.0, GPU job on `gpu_a100_short` ran (NVIDIA A100 40 GB); laptop is an Apple M1 with 8 GB (core agent only).
4. Plan approved by the user: minimal first agent with skills and analogy memory, defined metric, public repo. Repo published as `kiettruong-source/arc-agi-3-agent`; commits use a no-reply identity; no AI co-author line (user's request).
5. **v1** built and measured (E4). Information-gathering metrics added. A step-by-step trace exposed two bugs (F4, F5).
6. User: "for movement it should have a map in mind first, and plan the sequence of movement" → **v2** (E5). User: "make our skills general" → hard-coded assumptions removed.
7. User: analogy between levels by structure mapping → **v3**, plus the counter fix (E6).
8. Weakness diagnosis (E7). Experiment protocol written; AGI characteristics and task breakdown added on the user's request.
9. User: research the best technique and implement it → graph exploration, **v4** (E8): 3 → 7 dev levels.
10. Found that runs were not reproducible (F12); fixed; 3-seed baseline (E9): v4 6.67 ± 0.58 levels, random 1.33 ± 0.58.
11. Commits: `78b3ee6` v1 · `e35f06e` v2/v3 · `a7c2c29` protocol · `03008f5` AGI goal + tasks · `012beff` v4 · `3279f43` reproducibility + seeds.

### 2026-11-11 — Status review
1. Reviewed what went well and badly (summary in section 3 and protocol section 4). No code changes.

### 2026-11-12 — Backward reasoning, comparison with the best, hybrid decision
1. User asked for novel ideas with reverse thinking (from the goal back to the problem) and a focus on reasoning → six ideas (I24–I29). Read the game engine: the winning screen is rendered before the level switches, so the agent already receives it but ignores it.
2. User requirements: everything must be logged for the report (I30 → this logbook); the reasoning must be general (I31).
3. User asked whether the current best solutions are better than ours. Researched: yes, by far (protocol-style table in the plan; sources in the research notes). Ours: 0.15 / 100; Kaggle leaders ~55–75; frontier-LLM harnesses ~93–100 on the public games.
4. **Decision (user): hybrid** — a local LLM on bwUniCluster reasons, our core supplies evidence and checks, and plays alone without a GPU (I32).
5. Plan for v5 approved. Step 0 done: this logbook, `run.py --label` (commit `c73cb40`), baselines re-run as run files (`runs/2026-11-12-v4-baseline.json`, `…-random-baseline.json`); they reproduce E9 exactly (7, 7, 6 and 1, 2, 1 levels).
6. Step 1 done (E10): the winning screen is the second-to-last frame at every level-up seen; all game overs seen are step-limit timeouts at fixed intervals (31–132 actions per attempt). New understanding: a level must be solved inside one attempt, so game overs are a clock, not mistakes.

---

## 2. Ideas

Status: **kept** (in the agent / method) · **testing** · **proposed** (not built) · **parked** (later) · **changed** (replaced by another idea) · **rejected**.

| id | date | who | idea | status | evidence / reason |
|---|---|---|---|---|---|
| I1 | 11-04 | user | Focus on exploration that discovers the goal / end condition of an unknown game | kept | research question (protocol §1) |
| I2 | 11-04 | paper | LLM writes the game's rules as code, checks it against recordings, plans in it (Tycho, Kepler, Twin) | parked | strongest published approach; planned after the JSON-hypothesis version (I33) shows its limits |
| I3 | 11-04 | paper | Go-Explore: archive states, return to a promising one, explore from there | kept | realised as graph exploration (I22, E8) |
| I4 | 11-04 | Claude | Choose the action on which rival hypotheses disagree (query by disagreement) | proposed | not built; needs explicit hypotheses (I24–I26) |
| I5 | 11-04 | user | Cluster what is seen, like a human; test one member per cluster | kept | E1: 3–17 types instead of 4,096 clicks; E2: ~2× useful actions |
| I6 | 11-04 | user | See clusters as a map: identity (type) separate from position; track objects | kept | E3; perception in `agent.py` |
| I7 | 11-04 | user | Clusters get meaning after exploration (roles from observed effects) | kept, partly | roles = tries / effects / deaths; richer roles (wall, pushable, switch) not built |
| I8 | 11-04 | user | Reuse rule: similar cluster → one cheap check; new cluster → gather information | kept | E2: ~20% fewer probes, same effect |
| I9 | 11-04 | Claude | Never probe-confirm a type that was deadly before | kept, flawed | timeouts wrongly count as "deadly" (F16) |
| I10 | 11-04 | paper | Try small, rare-coloured objects first (salience) | kept | probe order; frontier tiers (E8) |
| I11 | 11-04 | Claude | Hold out 10 of 25 public games; tune only on 15 | kept | protocol §2 |
| I12 | 11-04 | Claude | Robustness test: colours permuted, grid flipped | proposed | task T7 |
| I13 | 11-08 | user | A cheap LLM adjusts the agent's settings; the core works without it | changed | became the hybrid design (I32) |
| I14 | 11-08 | user | The LLM should also see the screen (image + object list) | proposed | plan step 5 |
| I15 | 11-08 | Claude | Small set of fundamental skills; explore vs exploit decided by each skill's track record | kept | v1 (E4); weak once skills run out (F9) |
| I16 | 11-08 | user | Analogy memory / skill set shared across levels and games | kept | `Memory` |
| I17 | 11-08 | Claude | Store a win as a recipe of (action, object type) and replay it | kept, failing | 0 levels won in 9 replays (F10); to be limited to short recipes |
| I18 | 11-08 | user | Movement: build a map in mind first, then plan the sequence of moves | kept | E5: screens 682 → 1,254, wasted 63% → 38% |
| I19 | 11-08 | user | Skills must be general: reusable in unseen environments and with unseen actions | kept | no game ids, colours or action numbers in the code (E5) |
| I20 | 11-08 | user | Analogy between levels by structure mapping (relations over looks) | kept | mechanism works in a check (E6); no measured gain yet (F10) |
| I21 | 11-08 | Claude | Explicit goal-hypothesis space built from hand-written goal templates × roles | changed | hand-written templates dropped for a general contrast loop (I24, I31) |
| I22 | 11-08 | paper | Graph exploration: screens as nodes, shortest path to the nearest untried action | kept | E8: 3 → 7 dev levels; random actions 8,290 → 95 |
| I23 | 11-08 | Claude | Several seeds; levels as the primary number | kept | E9 |
| I24 | 11-12 | Claude | **Win snapshot:** what is true only on the winning screen = the goal | testing | E10: win screen is received as `frame[-2]` in 6 of 6 level-ups; plan step 2 |
| I25 | 11-12 | Claude | **"Not yet" evidence:** every non-winning screen is a free counter-example; look for new situations, not new screens | proposed | plan steps 2, later |
| I26 | 11-12 | Claude | **Means-ends analysis:** learn what each action type changes; pick the action that shrinks the difference to the goal | proposed | plan step 3 |
| I27 | 11-12 | Claude | **Transfer goals, not actions** to the next level | proposed | plan step 3; answer to F10 |
| I28 | 11-12 | Claude | **Deadline reasoning:** learn the step limit from deaths; timeout ≠ deadly object | proposed | E10: attempts last a fixed 31–132 actions; plan steps 3, 7; answer to F15, F16 |
| I29 | 11-12 | Claude | **Imagination:** predict an action's effect, search in the head, act only to test | parked | biggest effort; after the numbers of steps 2–6 |
| I30 | 11-12 | user | Log everything — ideas, failures, process — for the report | kept | this logbook, run records |
| I31 | 11-12 | user | The reasoning must be general | kept | one loop (event → contrast → hypothesis → test) for every game and question; plan Part B |
| I32 | 11-12 | user | **Hybrid:** local LLM on bwUniCluster reasons; the core supplies evidence, checks, and plays alone without a GPU | testing | plan steps 4–6 |
| I33 | 11-12 | Claude | The LLM answers as JSON expressions over the core's generic features; no LLM-written code is executed | proposed | safe and checkable; limits decide on I2 |
| I34 | 11-12 | paper | Reference run of the open "Duck" harness with the same model | parked | plan step 7 |
| I35 | 11-08 | Claude | Looser object identity (colour + size class) for shape-changing objects | proposed | answer to F13; task T3 |
| I36 | 11-04 | Claude | Small local web page to play the local environment in the browser | parked | offered, not needed so far |
| I37 | 11-12 | paper / Claude | Read all animation frames of an action, not only the last (Kepler: they carry task information) | proposed | E10: cd82, sp80, tn36, ls20 return 6–28 frames per action |

---

## 3. Failures and lessons

| id | date | what was tried | what happened | cause | what was done | lesson |
|---|---|---|---|---|---|---|
| F1 | 11-04 | `play.py`: `GameAction(1)` to look up an action by number | crash: "1 is not a valid GameAction" | the enum is not indexed by number | look up by name (`GameAction["ACTION1"]`) | test scripted input before handing over |
| F2 | 11-04 | E2 first run: "effective click" = any frame change | every click counted as effective; the cluster strategy kept clicking a shrinking bar | 4 of 7 click games draw a step counter that changes on every action | find the counter with a no-op click, mask it | effect detection must ignore counters first |
| F3 | 11-08 | v1 counter detection: any thin change at the border = counter | a real button near the border was masked; agent fell back to random after one probe | one change is not enough evidence | counter only after changing on two separate steps; the clicked object does not vote | found by the self-check, not by the benchmark |
| F4 | 11-08 | v1 on vc33 | the counter bar was clicked 13 times in a row | each time it shrank it had a new shape = "new type"; the rule from F3 protected it | bands that once changed on their own are never probed (v3) | a fix can open the opposite hole |
| F5 | 11-08 | v1 `go_to`: step greedily toward the target | ls20: 8 screens, 93% wasted actions; `go_to` used 8,361 of 15,000 actions at low value (0.14 per action) | oscillated against a wall (blocked, side-step, blocked…) | mental map + route planning (v2, user's idea I18) | plan on a map; do not react step by step |
| F6 | 11-08 | maze self-check for v2 | failed: only 2 of 4 moves learned, then random | the player's own movement was voted "counter" on a small grid | player's boxes excluded; margin scales with grid size | small synthetic worlds expose assumptions about 64×64 |
| F7 | 11-08 | v2 (map planning) | score fell 0.24 → 0.01 with the same number of levels (lost tn36, gained sp80) | not fully investigated (guess: changed counter detection altered the probe order in tn36) | recorded; v3 regained tn36 (slower) | the squared score hangs on single fast levels; one run proves little |
| F8 | 11-08 | `scripts/info_gathering.py` after changing `hud_bands()` | crash (`NoneType` not iterable) | script still used the old call | call fixed | helper scripts break silently when shared functions change |
| F9 | 11-08 | v3: random fallback when no skill applies | 8,997 of 15,000 actions random; click games: ~30–50 probes, then 900+ random | no goal hypotheses; nothing left to try | graph exploration (v4) | "what to do when out of ideas" decides most of the budget |
| F10 | 11-08 | replay of recipes on the next level (I17, I20) | 9 replay episodes, 0 levels won; up to 708 actions spent | a fixed action sequence does not fit a new layout, even mapped by role | open → transfer goals instead (I27) | transfer what was achieved, not what was pressed |
| F11 | 11-08 | v4 graph exploration | click coverage fell 74% → 64% | probing hands over to the graph search earlier | accepted (levels rose 3 → 7) | one metric can fall while the main one rises; report both |
| F12 | 11-08 | comparing versions with single runs | same v4 code gave 7 and 6 levels | Python's `hash()` is randomised per process (screen ids) | stable ids + fixed hash seed; 3 seeds (E9) | make runs reproducible before comparing anything; E4–E8 carry ~1 level of noise |
| F13 | 11-08 | object type = colour + exact shape | sk48: 896–987 probes on 154 "types"; tr87: 46 types | objects that change shape look new every time | open (I35) | exact matching is too strict for some games |
| F14 | 11-08 | `repeat` skill (v3) | ar25: 754 of 1,000 actions on `repeat` | it restarts on the same action with no memory | not fixed on purpose; in v4 it is no longer dominant (35 actions) | — |
| F15 | 11-08 | playing without regard to the step limit | 1–26 game overs per game; each costs the progress made | the counter is seen but not treated as time left; E10: they are timeouts at fixed intervals, and random has as many (~180 per run) as v4 | open (I28) | game overs are a clock: the question is what gets done inside one attempt |
| F16 | 11-12 | role "deadly" = the type clicked just before a game over | a timeout marks an innocent type deadly, which is then avoided | found by reading the code; E10: every game over seen was a timeout; effect on levels not yet measured | open → plan step 3 | check what a rule does in the case it was not written for |
| F17 | 11-08 | keeping run outputs only in a temporary folder | result files of E1–E9 cannot be shown; the numbers survive in the protocol | no saving step in the runner | `run.py --label` + `runs/` (2026-11-12); baselines re-run | evidence must be saved by the tool, not by hand |
| F18 | 11-08 | scoring by the official score alone | random's seed 1 scored 4.76 on one fast sp80 level and nearly matched v4's mean score | squared efficiency term | levels as the primary number, score alongside | — |
| F19 | 11-08 | v4 on the dev games | 9 of 15 games never complete a level (ar25, cn04, dc22, ft09, g50t, ka59, s5i5, sk48, tr87); 38–499 actions per completed level | exhaustive search without a goal | open → v5 | search finds levels; only understanding finds them cheaply |
| F20 | 11-08 | v4 on tu93 | 494 random actions; level 1 solved in only 2 of 3 seeds | graph found nothing reachable left (dead end) | open | — |
| F21 | 11-12 | reading the Kaggle first-place write-up | page could not be read by the tools | page needs a browser session | method recorded as unknown | state gaps in the comparison openly |
| F22 | 11-04 – 11-12 | small slips | macOS has no `timeout` command; a cluster command with a `<placeholder>` was run literally; a GPU test job was stopped with Ctrl+C and looked "FAILED"; the work e-mail was in `pyproject.toml` (caught before publishing); an AI co-author line was in the first commit (removed on request); experiment E8 was first placed above E7; a README line described replay the old way | — | each fixed when noticed | give copy-paste commands without placeholders; scan for personal data before publishing |
| F23 | 11-12 | run record flag "uncommitted changes" | the v4 baseline record says `true` although the code was identical to commit `c73cb40` | the flag also counted uncommitted documents | flag now checks code files only (`uncommitted_code_changes`) | a provenance flag must mean exactly one thing |

Open questions: Which goals do the games have (task R1)? Does the win-snapshot contrast predict the next level's win (goal rank)? How much does a ~27B local model add over the core alone?
