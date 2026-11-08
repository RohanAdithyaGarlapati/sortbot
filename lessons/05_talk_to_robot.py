"""Lesson 5: Talk to the robot.

Type a command in plain English; the robot plans and does it.

    you> sort the cubes by color
    you> build a tower with blue at the bottom and red on top
    you> put the green cube on the red pad, then go home

The "brain" turns text into a short list of Actions (place, sort, tower, home). By default it is
a set of keyword rules (sortbot/language.py). With --claude, Claude writes the plan instead, so
free-form requests such as "swap the red and blue cubes" work too.

Run it:
    python lessons/05_talk_to_robot.py                     # type commands; video saved when you quit
    python lessons/05_talk_to_robot.py --view              # live window while you type
    python lessons/05_talk_to_robot.py --commands "sort the cubes" "stack them"
    python lessons/05_talk_to_robot.py --claude            # needs: pip install anthropic
                                                           # and an ANTHROPIC_API_KEY
"""

import argparse
from pathlib import Path

from sortbot import Controller, Sim, Skills, execute, open_output, parse_command, plan_with_claude

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ["sort the cubes by color", "build a tower: blue, green, red",
            "put the red cube on the green pad", "stack blue on red", "go home"]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--view", action="store_true", help="open a live window instead of recording")
    parser.add_argument("--claude", action="store_true", help="let Claude plan instead of keyword rules")
    parser.add_argument("--commands", nargs="+", help="run these commands instead of asking")
    parser.add_argument("--shuffle", type=int, metavar="SEED", help="start the cubes in random spots")
    args = parser.parse_args()

    sim = Sim()
    if args.shuffle is not None:
        sim.reset(sim.random_layout(args.shuffle))
    out = open_output(sim.model, sim.data, ROOT / "videos" / "05_talk_to_robot.mp4", view=args.view)

    def say(message):
        print(f"robot> {message}")
        out.status = message

    skills = Skills(Controller(sim, on_step=out), say=say)

    def run(text):
        out.command = text
        if args.claude:
            actions, reply = plan_with_claude(text, skills.describe())
            say(reply)
        else:
            actions = parse_command(text)
        if not actions:
            say("Sorry, I didn't get that.")
            print("       Try: " + " | ".join(EXAMPLES))
            return
        print("plan>  " + ", ".join(str(a) for a in actions))
        execute(actions, skills)
        out.status = None

    try:
        if args.commands:
            for text in args.commands:
                print(f"you>   {text}")
                run(text)
        else:
            print("Type a command (or 'quit'). Examples: " + " | ".join(EXAMPLES))
            while (text := input("you>   ").strip()).lower() not in ("quit", "exit", "q"):
                if text:
                    run(text)
    except (KeyboardInterrupt, EOFError):
        print()
    finally:
        out.close()


if __name__ == "__main__":
    main()
