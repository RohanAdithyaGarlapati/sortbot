"""High-level skills: the only verbs the language layer is allowed to use.

Each skill looks at the world first (Is anything stacked on that cube? Is the pad taken?) and then
calls the Controller's pick and place primitives. Keeping this "is it safe to do?" logic in one place
means a parser, or an LLM, can only ask for things the robot knows how to do.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .motion import Controller
from .sim import COLORS, CUBE_HALF

# Free floor spots (x, y) on the robot's right where a cube can be set aside temporarily.
PARKING_SPOTS = [(0.42, -0.10), (0.56, -0.24), (0.66, -0.08), (0.40, -0.26), (0.66, -0.24), (0.52, -0.12)]


class Skills:
    def __init__(self, ctrl: Controller, say: Callable[[str], None] = print):
        self.ctrl = ctrl
        self.sim = ctrl.sim
        self.say = say   # where the robot's messages go (terminal, video caption, ...)

    # ------------------------------------------------------------------ looking at the world
    def resting_on(self, color: str) -> str | None:
        """The cube directly underneath ``color``, or None if it sits on the floor."""
        p = self.sim.cube_pos(color)
        for other in COLORS:
            if other != color:
                q = self.sim.cube_pos(other)
                if np.linalg.norm(p[:2] - q[:2]) < 0.035 and 0.035 < p[2] - q[2] < 0.065:
                    return other
        return None

    def on_top_of(self, color: str) -> str | None:
        """The cube sitting directly on ``color``, or None."""
        return next((c for c in COLORS if c != color and self.resting_on(c) == color), None)

    def occupant(self, pad: str) -> str | None:
        """The cube standing on the floor at this pad, or None."""
        return next((c for c in COLORS if self.sim.cube_on_pad(c, pad, tol=0.045)), None)

    def free_spot(self) -> tuple[float, float] | None:
        cubes = [self.sim.cube_pos(c)[:2] for c in COLORS]
        for spot in PARKING_SPOTS:
            if all(np.linalg.norm(np.array(spot) - c) > 0.08 for c in cubes):
                return spot
        return None

    def describe(self) -> str:
        """Plain-English summary of the scene (also handed to the LLM in Claude mode)."""
        lines = []
        for c in COLORS:
            below = self.resting_on(c)
            pad = next((p for p in COLORS if self.sim.cube_on_pad(c, p)), None)
            if below:
                where = f"stacked on the {below} cube"
            elif pad:
                where = f"on the {pad} pad"
            else:
                x, y, _ = self.sim.cube_pos(c)
                where = f"on the floor at x={x:.2f} m, y={y:.2f} m"
            lines.append(f"- {c} cube: {where}")
        return "\n".join(lines)

    # ------------------------------------------------------------------ skills
    def place(self, obj: str, target: str) -> bool:
        """Put cube ``obj`` on ``target``, which is "<color>_pad" or "<color>_cube"."""
        tcolor, kind = target.split("_")
        if kind == "cube" and tcolor == obj:
            self.say(f"I can't put the {obj} cube on itself.")
            return False
        if kind == "cube" and self.resting_on(obj) == tcolor:
            self.say(f"The {obj} cube is already on the {tcolor} cube.")
            return True
        if kind == "pad" and self.sim.cube_on_pad(obj, tcolor):
            self.say(f"The {obj} cube is already on the {tcolor} pad.")
            return True

        self._clear_top(obj)                         # 1. nothing may sit on the cube we pick up
        if kind == "cube":
            self._clear_top(tcolor)                  # 2a. the cube we stack on must be clear
            dest = self.sim.cube_pos(tcolor) + [0.0, 0.0, 2 * CUBE_HALF]
        else:
            occ = self.occupant(tcolor)              # 2b. the pad must be empty
            if occ and occ != obj:
                self._clear_top(occ)
                self._set_aside(occ)
            pad = self.sim.pad_pos(tcolor)
            dest = np.array([pad[0], pad[1], CUBE_HALF])
        return self._pick_and_place(obj, dest, f"onto the {tcolor} {kind}")

    def sort(self) -> bool:
        """Put every cube on the pad with the same color."""
        for _ in range(8):
            todo = [c for c in COLORS if not self.sim.cube_on_pad(c)]
            if not todo:
                self.say("All cubes are on their matching pads.")
                return True
            # Prefer a cube that is free to grab and whose pad is empty: no extra moves needed.
            easy = [c for c in todo if not self.on_top_of(c) and not self.occupant(c)]
            self.place((easy or todo)[0], f"{(easy or todo)[0]}_pad")
        return all(self.sim.cube_on_pad(c) for c in COLORS)

    def tower(self, order: list[str]) -> bool:
        """Stack cubes into a tower, ``order`` listed from bottom to top."""
        base = order[0]
        if self.resting_on(base):                    # the base has to stand on the floor
            self._clear_top(base)
            self._set_aside(base)
        for lower, upper in zip(order, order[1:]):
            if self.resting_on(upper) != lower:
                if not self.place(upper, f"{lower}_cube"):
                    return False
        self.say("Tower built: " + ", ".join(order) + " (bottom to top).")
        return True

    def home(self) -> bool:
        self.ctrl.go_home()
        return True

    # ------------------------------------------------------------------ helpers
    def _pick_and_place(self, obj: str, dest, label: str) -> bool:
        self.say(f"Moving the {obj} cube {label}.")
        for attempt in range(2):
            if self.ctrl.pick(obj):
                self.ctrl.place(dest, label)
                return True
            self.say("Missed the grasp, trying again." if attempt == 0 else "I couldn't get a grip on it.")
        return False

    def _set_aside(self, color: str) -> None:
        spot = self.free_spot()
        if spot is None:
            raise RuntimeError("No free floor space left to set a cube aside.")
        self._pick_and_place(color, np.array([spot[0], spot[1], CUBE_HALF]), "out of the way")

    def _clear_top(self, color: str) -> None:
        while (top := self.on_top_of(color)) is not None:
            self._clear_top(top)
            self._set_aside(top)
