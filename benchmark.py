"""Run many simulated trials without video and log every command to a CSV for analysis.

Each trial starts the cubes in a different random layout and runs the same five commands.
After every action we check the world ourselves (is the cube really on the pad? is the
tower really stacked?) instead of trusting the robot's own report.

    python benchmark.py                 # 30 trials -> logs/trials.csv
    python benchmark.py --trials 100

Open logs/trials.csv in Power BI or Excel to chart success rate and cycle time.
"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from sortbot import COLORS, Controller, Sim, Skills, parse_command

ROOT = Path(__file__).resolve().parent
COMMANDS = [
    "sort the cubes by color",
    "build a tower: blue, green, red",
    "make a tower with red at the bottom",
    "put the red cube on the blue pad and blue on the red pad",
    "sort the cubes by color",
]
COLUMNS = ["run_id", "trial", "step", "command", "actions", "success", "sim_seconds", "wall_seconds",
           "cube_moves", "grasp_retries", "start_distance_m", "error"]


def action_done(action, sim, skills) -> bool:
    """Ground truth: look at where the cubes actually are."""
    if action.kind == "sort":
        return all(sim.cube_on_pad(c) for c in COLORS)
    if action.kind == "tower":
        return all(skills.resting_on(upper) == lower for lower, upper in zip(action.order, action.order[1:]))
    if action.kind == "place":
        color, kind = action.target.split("_")
        return sim.cube_on_pad(action.obj, color) if kind == "pad" else skills.resting_on(action.obj) == color
    return True   # "home" has nothing to check


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--out", default=str(ROOT / "logs" / "trials.csv"))
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out.exists()
    run_id = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows = []

    with open(out, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()

        for trial in range(args.trials):
            sim = Sim()
            sim.reset(sim.random_layout(trial))
            start_distance = sum(np.linalg.norm(sim.cube_pos(c)[:2] - sim.pad_pos(c)[:2]) for c in COLORS)
            messages = []
            skills = Skills(Controller(sim), say=messages.append)

            for step, command in enumerate(COMMANDS, start=1):
                messages.clear()
                actions = parse_command(command)
                sim_start, wall_start = sim.time, time.perf_counter()
                success, error = True, ""
                try:
                    for action in actions:
                        if action.kind == "place":
                            skills.place(action.obj, action.target)
                        elif action.kind == "sort":
                            skills.sort()
                        elif action.kind == "tower":
                            skills.tower(action.order)
                        else:
                            skills.home()
                        success = success and action_done(action, sim, skills)
                except Exception as exc:          # log it and keep going
                    success, error = False, str(exc)

                row = {
                    "run_id": run_id, "trial": trial, "step": step, "command": command,
                    "actions": len(actions), "success": success,
                    "sim_seconds": round(sim.time - sim_start, 2),
                    "wall_seconds": round(time.perf_counter() - wall_start, 2),
                    "cube_moves": sum(m.startswith("Moving the") for m in messages),
                    "grasp_retries": sum(m.startswith("Missed the grasp") for m in messages),
                    "start_distance_m": round(start_distance, 3),
                    "error": error,
                }
                writer.writerow(row)
                rows.append(row)
            print(f"trial {trial + 1}/{args.trials} done")

    # Quick summary in the terminal
    print(f"\nSaved {len(rows)} rows to {out}\n")
    print(f"{'step':<5} {'command':<58} {'success':>8} {'avg sim s':>10}")
    for step, command in enumerate(COMMANDS, start=1):
        subset = [r for r in rows if r["step"] == step]
        rate = 100 * sum(r["success"] for r in subset) / len(subset)
        secs = sum(r["sim_seconds"] for r in subset) / len(subset)
        print(f"{step:<5} {command:<58} {rate:>7.0f}% {secs:>10.1f}")


if __name__ == "__main__":
    main()
