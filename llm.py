"""Optional LLM reasoner: one call to any OpenAI-compatible endpoint (vLLM on a cluster, Ollama, a hosted API).

Settings (environment / .env):
  LLM_BASE_URL   e.g. http://localhost:8000/v1   -- unset = no LLM, the core agent plays alone
  LLM_MODEL      model name the server expects
  LLM_API_KEY    optional
  LLM_EXTRA      optional JSON merged into the request, e.g. '{"top_k": 20}'

The LLM is never trusted: `ask()` returns text or None, and every hypothesis it proposes is checked by the core
against the recorded evidence before it is used.
"""

import base64
import io
import json
import os

import requests

CALLS = {"asked": 0, "answered": 0}  # for the run record


def enabled():
    return bool(os.environ.get("LLM_BASE_URL"))


def ask(prompt, image=None, max_tokens=4000, timeout=180):
    """Send one question (optionally with a PNG image). Any failure -> None; the caller carries on without it."""
    base = os.environ.get("LLM_BASE_URL")
    if not base:
        return None
    content = [{"type": "text", "text": prompt}]
    if image is not None:
        url = "data:image/png;base64," + base64.b64encode(image).decode()
        content.insert(0, {"type": "image_url", "image_url": {"url": url}})
    body = {"model": os.environ.get("LLM_MODEL", ""), "messages": [{"role": "user", "content": content}],
            "max_tokens": max_tokens, "temperature": 0.6}
    CALLS["asked"] += 1
    try:
        body.update(json.loads(os.environ.get("LLM_EXTRA") or "{}"))
        r = requests.post(base.rstrip("/") + "/chat/completions", json=body, timeout=timeout,
                          headers={"Authorization": "Bearer " + os.environ.get("LLM_API_KEY", "EMPTY")})
        r.raise_for_status()
        text = r.json()["choices"][0]["message"]["content"]
    except (requests.RequestException, KeyError, IndexError, TypeError, ValueError):
        return None
    CALLS["answered"] += bool(text)
    return text or None


def render(grid, scale=8):
    """The screen as PNG bytes, in the benchmark's own colours."""
    import numpy as np
    from arc_agi.rendering import COLOR_MAP
    from PIL import Image

    pal = np.array([[int(COLOR_MAP.get(i, "#000000FF")[j:j + 2], 16) for j in (1, 3, 5)] for i in range(16)], np.uint8)
    rgb = pal[np.clip(np.asarray(grid), 0, 15)].repeat(scale, 0).repeat(scale, 1)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, "PNG")
    return buf.getvalue()


def parse_json(text):
    """The first JSON object in an answer (models wrap JSON in prose or code fences). None if there is none."""
    if not text:
        return None
    start, end = text.find("{"), text.rfind("}")
    try:
        return json.loads(text[start:end + 1]) if 0 <= start < end else None
    except ValueError:
        return None


PROMPT = """You are the reasoning part of an agent that plays an unknown grid game. Nobody has told the agent the rules or the goal. A level ends the moment a hidden goal condition becomes true. Work out what that condition most likely is.

The image shows the current screen. Facts measured by the agent:
{facts}

Describe a goal only with these measures:
- "count": how many separate objects of a colour      - "cells": how many cells of a colour
- "dist": gap, in steps, between the thing the agent controls and the nearest object of a colour (0 = touching)
- "objects", "types", "colours", "odd": totals for the whole screen (objects, different shapes, different colours, colours that occur exactly once); these take no colour

Answer with JSON only, most likely goal first, at most 5 goals:
{{"goals": [{{"why": "<one sentence>", "all": [{{"measure": "<measure>", "colour": <colour number>, "op": "<== or <= or >=>", "value": <number>}}]}}]}}
Use only colours from the table. A goal must not already be true on a screen where the level did not end (see the values seen so far)."""


def propose(agent, start, seen):
    """Ask for goal hypotheses about the agent's current level. Returns [(why, conditions), ...] (validated form
    only; checking them against the evidence is the core's job). The prompt is the same for every game."""
    from agent import CLICK, conditions, schema_text, val

    now = agent.feat(agent.grid)
    colours = sorted(k[2] for k in now if isinstance(k, tuple) and k[:2] == ("count", "colour"))
    name = lambda k: " ".join(str(x) for x in k) if isinstance(k, tuple) else k
    lines = [f"Grid {agent.grid.shape[0]}x{agent.grid.shape[1]}. Level {agent.level + 1} of this game.",
             "colour | objects | cells | steps to the controlled thing"]
    lines += [f"{c} | {now[('count', 'colour', c)]} | {now[('cells', 'colour', c)]} | {now.get(('dist', 'colour', c), '-')}"
              for c in colours]
    mine = sorted({t[0] for t in agent.player})
    lines.append(f"The agent controls: {'the object(s) of colour ' + str(mine) if mine else 'nothing found yet'}.")
    moves = {a: v for a, v in agent.steps().items()}
    lines.append(f"Actions: {sorted(agent.actions)}" + (f"; action -> (dx, dy) of the controlled thing: {moves}" if moves else "")
                 + ("; action 6 is a click on a cell." if CLICK in agent.actions else ""))
    effects = []
    for (act, typ), (delta, n) in agent.mem.ops.items():
        top = sorted(delta.items(), key=lambda kv: -abs(kv[1]))[:3]
        if n and top and (typ is None or typ[0] in colours):
            what = f"click on a colour-{typ[0]} object" if typ else f"action {act}"
            effects.append(f"- {what}: tried {n}x; average change: " + ", ".join(f"{name(k)} {v / n:+.1f}" for k, v in top))
    lines += ["What actions changed so far:"] + (effects[:12] or ["- nothing recorded yet"])
    varying = []
    for k in sorted({k for f in seen for k in f}, key=str):
        vs = [v for v in (val(f, k) for f in seen) if v is not None]
        if vs and min(vs) != max(vs):
            varying.append(f"{name(k)}: {min(vs)}..{max(vs)}")
    lines.append(f"The level did NOT end on {len(seen)} screens. Values seen on them (others never changed): "
                 + ("; ".join(varying[:30]) or "nothing varied"))
    if agent.goal:
        lines.append("Earlier level of this game was won when: " + "; ".join(schema_text(agent.goal[0])))
    if agent.refuted:
        lines.append("Already tested and wrong on this level: " + "; ".join(agent.refuted))
    answer = parse_json(ask(PROMPT.format(facts="\n".join(lines)), render(agent.grid)))
    out = []
    for g in (answer or {}).get("goals", [])[:5] if isinstance(answer, dict) else []:
        conds = conditions(g)
        if conds:
            out.append((str(g.get("why", ""))[:200], conds))
    return out
