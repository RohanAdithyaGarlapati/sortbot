"""Lesson 4: Pick and place.

Now we put lessons 1 to 3 together. ``Controller`` (sortbot/motion.py) runs the same IK from
lesson 3 every physics step and glides the gripper along smooth paths. On top of it, a
pick-and-place is just a fixed sequence of moves, a tiny "state machine":

    hover above the cube -> descend -> close gripper -> lift ->
    carry over the pad   -> lower   -> open gripper  -> back off

Run it:
    python lessons/04_pick_and_place.py              # records videos/04_pick_and_place.mp4
    python lessons/04_pick_and_place.py --view
    python lessons/04_pick_and_place.py --shuffle 7  # random cube positions (any number works)
"""

import argparse
from pathlib import Path

from sortbot.motion import TRAVEL_Z, Controller
from sortbot.sim import COLORS, CUBE_HALF, Sim
from sortbot.video import open_output

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--view", action="store_true", help="open a live window instead of recording")
    parser.add_argument("--shuffle", type=int, metavar="SEED", help="start the cubes in random spots")
    args = parser.parse_args()

    sim = Sim()
    if args.shuffle is not None:
        sim.reset(sim.random_layout(args.shuffle))
    out = open_output(sim.model, sim.data, ROOT / "videos" / "04_pick_and_place.mp4", view=args.view)
    ctrl = Controller(sim, on_step=out)   # out() runs after every physics step

    def step(text, action, *args, **kwargs):
        out.status = text
        action(*args, **kwargs)

    def pick_and_place(color):
        cube, pad = sim.cube_pos(color), sim.pad_pos(color)
        step(f"{color}: hover above the cube", ctrl.move_to, [cube[0], cube[1], TRAVEL_Z])
        step(f"{color}: descend", ctrl.move_to, cube, speed=0.25)
        step(f"{color}: close the gripper", ctrl.close_gripper)
        if not ctrl.holding_something():
            print(f"Missed the {color} cube!")
            return
        step(f"{color}: lift", ctrl.move_to, [cube[0], cube[1], TRAVEL_Z], speed=0.3)
        step(f"{color}: carry it to the {color} pad", ctrl.move_to, [pad[0], pad[1], TRAVEL_Z])
        step(f"{color}: lower", ctrl.move_to, [pad[0], pad[1], CUBE_HALF + 0.004], speed=0.25)
        step(f"{color}: open the gripper", ctrl.open_gripper)
        step(f"{color}: back off", ctrl.move_to, [pad[0], pad[1], TRAVEL_Z], speed=0.3)

    for color in COLORS:
        pick_and_place(color)
    step("done: going home", ctrl.go_home)
    out.close()

    for color in COLORS:
        offset_mm = 1000 * abs(sim.cube_pos(color)[:2] - sim.pad_pos(color)[:2]).max()
        result = "OK " if sim.cube_on_pad(color) else "MISS"
        print(f"{result} {color:<5} cube is {offset_mm:4.1f} mm from the center of its pad")
    print(f"Finished in {sim.time:.1f} simulated seconds.")


if __name__ == "__main__":
    main()
