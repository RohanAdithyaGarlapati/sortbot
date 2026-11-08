"""Turn plain English into robot actions.

Two interchangeable "brains":

* ``parse_command``      simple keyword rules. Free, offline, predictable.
* ``plan_with_claude``   asks Claude (Anthropic API) to write the plan. Handles free-form requests
                         like "swap the red and blue cubes". Needs ``pip install anthropic`` and an
                         ANTHROPIC_API_KEY environment variable.

Both return the same thing: a list of ``Action`` objects. The robot only ever executes Actions,
so whichever brain you use, it can only ask for moves the robot knows how to do safely.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .sim import COLORS

PAD_WORDS = ("pad", "square", "zone", "spot", "tile", "mat", "area")
CUBE_WORDS = ("cube", "block", "box")
SORT_WORDS = ("sort", "match", "tidy", "organize", "organise", "clean up", "by color", "by colour")
TOWER_WORDS = ("tower", "stack them", "stack all", "stack everything", "pile", "stack the cubes")
HOME_WORDS = ("home", "reset", "rest", "go back", "start position")
DEFAULT_TOWER = ["blue", "green", "red"]  # bottom to top


@dataclass
class Action:
    kind: str                      # "place", "sort", "tower" or "home"
    obj: str | None = None         # place: which cube moves
    target: str | None = None      # place: "<color>_pad" or "<color>_cube"
    order: list[str] = field(default_factory=list)  # tower: colors from bottom to top

    def __str__(self) -> str:
        if self.kind == "place":
            return f"place({self.obj} -> {self.target})"
        if self.kind == "tower":
            return f"tower({', '.join(self.order)})"
        return f"{self.kind}()"


# ---------------------------------------------------------------------- rule-based parser
def _colors_in(text: str) -> list[tuple[str, int]]:
    """Colors in the order they appear, with their positions in the text."""
    return [(m.group(), m.start()) for m in re.finditer(r"\b(red|green|blue)\b", text)]


def _tower_order(clause: str) -> list[str]:
    """Bottom-to-top order for a tower. Towers always use all three cubes.

    Colors are taken in the order mentioned; "X on top" and "X at the bottom" move X;
    colors not mentioned go in the middle.
    """
    found = _colors_in(clause)
    mentioned, top, bottom = [], None, None
    for i, (color, pos) in enumerate(found):
        if color not in mentioned:
            mentioned.append(color)
        end = found[i + 1][1] if i + 1 < len(found) else len(clause)
        words_after = clause[pos:end]   # text up to the next color only
        if re.search(r"\b(on top|at the top|on the top|last)\b", words_after):
            top = color
        elif re.search(r"\b(at the bottom|on the bottom|first|as the base)\b", words_after):
            bottom = color
    middle = [c for c in mentioned if c not in (top, bottom)]
    middle += [c for c in DEFAULT_TOWER if c not in mentioned]
    return ([bottom] if bottom else []) + middle + ([top] if top else [])


def _parse_place(clause: str) -> Action | None:
    colors = _colors_in(clause)
    if len(colors) < 2:
        return None
    (obj, _), (tcolor, tpos) = colors[0], colors[1]
    after = clause[tpos + len(tcolor):tpos + len(tcolor) + 12]
    if any(w in after for w in PAD_WORDS):
        kind = "pad"
    elif any(w in after for w in CUBE_WORDS) or "stack" in clause or "top of" in clause:
        kind = "cube"
    else:
        kind = "pad"
    return Action("place", obj=obj, target=f"{tcolor}_{kind}")


def _is_command(clause: str) -> bool:
    words = SORT_WORDS + TOWER_WORDS + ("put", "move", "place", "stack", "go ", "home", "reset")
    return any(w in clause for w in words)


def parse_command(text: str) -> list[Action]:
    """Keyword rules. Examples it understands:

        "sort the cubes by color"
        "put the red cube on the blue pad"
        "stack green on red, then go home"
        "build a tower: blue, green, red"     (bottom to top)
        "make a tower with red on top"
    """
    actions: list[Action] = []
    clauses = [c.strip(" ,:") for c in re.split(r"\bthen\b|[;.!?]", text.lower())]
    clauses = [c for c in clauses if c]
    i = 0
    while i < len(clauses):
        clause = clauses[i]
        i += 1
        if any(w in clause for w in TOWER_WORDS):
            # "tower: blue, then green, then red": keep absorbing clauses that only name one color.
            while i < len(clauses) and len(_colors_in(clauses[i])) == 1 and not _is_command(clauses[i]):
                clause += ", " + clauses[i]
                i += 1
            actions.append(Action("tower", order=_tower_order(clause)))
        elif any(w in clause for w in SORT_WORDS):
            actions.append(Action("sort"))
        elif any(re.search(rf"\b{w}\b", clause) for w in HOME_WORDS) and len(_colors_in(clause)) < 2:
            actions.append(Action("home"))
        else:
            # Possibly several placements: "red on the blue pad and green on the red pad"
            for part in re.split(r",|\band\b", clause):
                action = _parse_place(part)
                if action:
                    actions.append(action)
    return actions


# ---------------------------------------------------------------------- Claude as the planner
PLAN_TOOL = {
    "name": "run_plan",
    "description": "Send the robot an ordered list of steps to execute, plus a one-sentence reply for the user.",
    "input_schema": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "action": {"type": "string", "enum": ["place", "sort", "tower", "home"]},
                        "object": {"type": "string", "enum": list(COLORS),
                                   "description": "place only: the cube to move"},
                        "target": {"type": "string",
                                   "enum": [f"{c}_pad" for c in COLORS] + [f"{c}_cube" for c in COLORS],
                                   "description": "place only: a pad, or a cube to stack on"},
                        "order": {"type": "array", "items": {"type": "string", "enum": list(COLORS)},
                                  "description": "tower only: colors from bottom to top"},
                    },
                    "required": ["action"],
                },
            },
            "reply": {"type": "string", "description": "Short sentence for the user: what you will do, or why you can't."},
        },
        "required": ["steps", "reply"],
    },
}

SYSTEM_PROMPT = """You plan actions for a simulated robot arm (a Franka Panda) on a floor with three \
5 cm cubes (red, green, blue) and three flat colored pads (red, green, blue).

The robot can only do these steps:
- place: pick up one cube and put it on a pad or on top of another cube. If the destination is \
occupied, or something is stacked on the cube, the robot sets the blocking cube aside on its own.
- sort: put every cube on the pad of its own color.
- tower: stack cubes into one tower, given bottom to top.
- home: move the arm back to its rest pose.

Always answer by calling run_plan. Steps run in order. Keep plans short. If the request is impossible \
or unrelated to the cubes, return no steps and explain briefly in the reply."""


def plan_with_claude(text: str, scene_description: str, model: str = "claude-haiku-5-5") -> tuple[list[Action], str]:
    """Ask Claude to turn a request into Actions. Returns (actions, reply)."""
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError("Claude mode needs the SDK: pip install anthropic") from exc

    client = anthropic.Anthropic()  # reads the ANTHROPIC_API_KEY environment variable
    response = client.messages.create(
        model=model,
        max_tokens=1024,
        system=SYSTEM_PROMPT + "\n\nCurrent scene:\n" + scene_description,
        tools=[PLAN_TOOL],
        tool_choice={"type": "tool", "name": "run_plan"},   # force a structured answer
        messages=[{"role": "user", "content": text}],
    )
    plan = next(block.input for block in response.content if block.type == "tool_use")
    return actions_from_plan(plan), plan.get("reply", "")


def actions_from_plan(plan: dict) -> list[Action]:
    """Validate the plan an LLM returned. Anything malformed is dropped, never executed."""
    valid_targets = {f"{c}_{k}" for c in COLORS for k in ("pad", "cube")}
    actions = []
    for step in plan.get("steps", []):
        kind = step.get("action")
        if kind == "place" and step.get("object") in COLORS and step.get("target") in valid_targets:
            actions.append(Action("place", obj=step["object"], target=step["target"]))
        elif kind == "tower":
            order = [c for c in step.get("order", []) if c in COLORS]
            if len(order) >= 2 and len(set(order)) == len(order):
                actions.append(Action("tower", order=order))
        elif kind in ("sort", "home"):
            actions.append(Action(kind))
    return actions


# ---------------------------------------------------------------------- running actions
def execute(actions: list[Action], skills) -> None:
    """Run Actions one by one with the robot's skills."""
    for action in actions:
        if action.kind == "place":
            skills.place(action.obj, action.target)
        elif action.kind == "sort":
            skills.sort()
        elif action.kind == "tower":
            skills.tower(action.order)
        elif action.kind == "home":
            skills.home()
