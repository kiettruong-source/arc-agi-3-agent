# arc-agi-3-agent

A minimal agent for [ARC-AGI-3](https://arcprize.org/) interactive games (university project, WS 26/27).
It discovers each game's rules and goal by **cheap exploration**, groups what it sees into **object types**,
keeps an **analogy memory** of what each type did, and switches between **exploring** (gathering
information) and **exploiting** (using a fundamental skill or a remembered recipe).

## Setup

```bash
uv sync
cp .env.example .env          # optional: ARC_API_KEY (an anonymous key is used otherwise)
uv run play.py ls20           # play by hand: wasd, e, 6 X Y to click, u undo, r reset, q quit
uv run run.py dev             # evaluate the agent on the dev games
uv run run.py dev --agent random   # baseline
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

3. **Explore vs exploit**: replay first if a recipe matches; otherwise explore while unknown things remain,
   unless an exploit skill's track record (value per step) is better. Random fallback when nothing applies.
4. **Mental map** (movement games): floor colours learned from where the player steps, walls from blocked moves,
   places visited; routes planned by breadth-first search. Any action that moves the player counts as a move.
5. **Analogy memory** (`Memory`), shared across levels and games: per-type roles (tries / effects / deaths),
   winning recipes (shortest known path), per-skill value. Recipes store each target's **relational description**
   (odd one out, how many alike, relative size, side of the player) and are mapped onto new levels by
   **structure mapping**: relations weigh more than colour / shape.

## Evaluation metric

The toolkit's official scorecard (RHAE-style):

- per level: `min((human_baseline_actions / agent_actions)² × 100, 115)`; 0 if the level is not completed
- per game: level scores averaged with weight = level index (later levels count more), capped at 100
- overall: mean over games; also reported: levels completed, actions

Protocol: tune only on the 15 **dev** games; the 10 **held-out** games (`run.py heldout`) are for final
numbers. Fixed budget of 1000 actions per game, always compared with the random baseline.

## Repo layout

- `agent.py`: agent, skills, memory · `run.py`: evaluation · `play.py`: play by hand · `test_agent.py`: self-check
- `scripts/`: exploration analysis (object clustering, object map)
- `docs/protocol.md`: experiment protocol (setup, metric, experiments E1-E7, decisions, bugs, next steps)
- `docs/research/`: research notes (landscape, Tycho, Go-Explore, clustering experiments)

Game source files downloaded by the toolkit (`environment_files/`) are not part of this repo and the agent never reads them.
