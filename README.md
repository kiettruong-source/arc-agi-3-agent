# arc-agi-3-agent

A minimal agent for [ARC-AGI-3](https://arcprize.org/) interactive games (university project, WS 26/27).
It discovers each game's rules and goal by **cheap exploration**, groups what it sees into **object types**,
keeps an **analogy memory** of what each type did, switches between **exploring** (gathering
information) and **exploiting**, and **reasons backward from the goal**: it learns what the end of a level
looks like from the winning screen and, optionally, lets a local LLM propose goals that the core then checks.

## Setup

```bash
uv sync
cp .env.example .env          # optional: ARC_API_KEY (an anonymous key is used otherwise)
uv run play.py ls20           # play by hand: wasd, e, 6 X Y to click, u undo, r reset, q quit
uv run run.py dev --seeds 3 --label NAME   # evaluate on the dev games; --label saves the result to runs/
uv run run.py dev --seeds 3 --clean-run    # also replay the found solutions in a fresh play ("best play" number)
uv run python scripts/score_anatomy.py runs/FILE.json   # where the score points come from
uv run run.py dev --seeds 3 --agent random   # baseline
uv run python test_agent.py   # self-check
uv run python scripts/info_gathering.py   # exploration metrics vs random
```

## How the agent works (`agent.py`)

1. **Perceive**: frame → same-colour connected objects; type = colour + shape (position-free).
   Counters / bars along the border are masked as HUD once they change on two separate steps.
2. **Fundamental skills** (built-in instinct, from core-knowledge priors):

   | skill | kind | idea |
   |---|---|---|
   | `probe` | explore | act once per untested object type (rare / small first); learn what each move does |
   | `replay` | exploit | replay a winning recipe from memory, mapped onto the objects that play the same role (structure mapping) |
   | `repeat` | exploit | keep doing what just changed the world |
   | `click_all` | exploit | click every object of a type that did something |
   | `go_to` | exploit | plan a route on the mental map to each untried object type |
   | `explore_map` | explore | plan a route to the nearest place the player has never been |
   | `frontier` | explore | graph exploration: untried action here, else shortest known path to the nearest screen with one (small objects first) |

3. **Explore vs exploit**: replay first if a recipe matches; otherwise explore while unknown things remain,
   unless an exploit skill's track record (value per step) is better. When no skill applies: graph exploration (`frontier`), random only as a last resort.
4. **Mental map** (movement games): floor colours learned from where the player steps, walls from blocked moves,
   places visited; routes planned by breadth-first search. Any action that moves the player counts as a move.
5. **Analogy memory** (`Memory`), shared across levels and games: per-type roles (tries / effects / deaths),
   winning recipes (shortest known path), per-skill value. Recipes store each target's **relational description**
   (odd one out, how many alike, relative size, side of the player) and are mapped onto new levels by
   **structure mapping**: relations weigh more than colour / shape.
6. **Goal reasoning, backward from the end** — one general loop: *event → contrast → hypothesis → test → revise*.
   - Evidence: every screen where the level did not end is a free counter-example; the **winning screen**
     (the engine sends it just before the next level) is the positive example.
   - A goal is whatever distinguishes the winning screen, described with generic features (per colour / size
     class: how many objects, how many cells, distance to the controlled thing; plus totals). No goal types are
     written by hand.
   - The goal — not the action sequence — is carried to the next level and pursued first: the search becomes
     best-first, actions are ordered by their known effect (means-ends analysis), `go_to` walks to the goal's
     colours. It is dropped when it stops making progress.
   - Switches, off by default because they did not help on the dev games (E12): treat a game over at the step
     limit as a timeout instead of a deadly object (`deadline`); replay only short recipes (`replay5`).
7. **Optional LLM reasoner** (`llm.py`, off unless `LLM_BASE_URL` is set): a local vision model proposes goal
   hypotheses *before the first win*, as conditions over the same generic features. They are data, not code,
   and never trusted: the core rejects what the evidence already contradicts and refutes the rest in play.
   Scripts for a GPU cluster are in `cluster/`.

Current state (dev games, 3 seeds): see `docs/protocol.md`. The goal reasoning works on toy worlds but has not
yet improved the dev-game result (E12, a negative result); the LLM reasoner has not yet been run with a real model (E13).

## Evaluation metric

The toolkit's official scorecard (RHAE-style):

- per level: `min((human_baseline_actions / agent_actions)² × 100, 115)`; 0 if the level is not completed
- per game: level scores averaged with weight = level index (later levels count more), capped at 100
- overall: mean over games; also reported: levels completed, actions

Two ways of counting, always reported separately:
- **first exposure** (main number): the first play of each game, every action counts, exploring included;
- **best play** (`--clean-run`): after the first play the agent replays the shortest solution it found in a fresh
  play; the toolkit scores the best play of a game. It shows how short the found solutions are, not how
  efficiently the agent learns.

Protocol: tune only on the 15 **dev** games; the 10 **held-out** games (`run.py heldout`) are for final
numbers. Fixed budget of 1000 actions per game, at least 3 seeds, always compared with the random baseline.
Levels completed is the primary comparison number (the squared score is dominated by single fast levels).

## Repo layout

- `agent.py`: agent, skills, memory, goal reasoning · `llm.py`: optional LLM reasoner · `run.py`: evaluation · `play.py`: play by hand · `test_agent.py`: self-check
- `cluster/`: setup and job script for running the LLM reasoner on a GPU cluster (untested)
- `scripts/`: analysis scripts (object clustering, object map, exploration metrics, win-frame probe)
- `docs/protocol.md`: experiment protocol (setup, metric, experiments, decisions, bugs, tasks)
- `docs/logbook.md`: project logbook (process journal, every idea, every failure and lesson)
- `runs/`: saved results of every evaluation (`run.py --label NAME`)
- `docs/research/`: research notes (landscape, Tycho, Go-Explore, clustering experiments)

Game source files downloaded by the toolkit (`environment_files/`) are not part of this repo and the agent never reads them.
