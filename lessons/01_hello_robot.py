"""Lesson 1: Meet the robot.

Goal: load a robot arm into a physics simulator, see what it is made of, and make it move.

Run it:
    python lessons/01_hello_robot.py           # records videos/01_hello_robot.mp4
    python lessons/01_hello_robot.py --view    # live 3D window (on a Mac use: mjpython ...)

Key ideas:
    model  = the fixed description of the world (from the XML file)
    data   = the changing state: joint angles, velocities, time
    ctrl   = what you ask the motors to do
    qpos   = where the joints actually are
    mj_step(model, data) advances physics by one tiny timestep
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from sortbot.video import open_output

ROOT = Path(__file__).resolve().parent.parent


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--view", action="store_true", help="open a live window instead of recording")
    args = parser.parse_args()

    # 1) Load the world. The XML describes every body, joint, motor, light and color.
    model = mujoco.MjModel.from_xml_path(str(ROOT / "robot" / "scene.xml"))
    data = mujoco.MjData(model)

    # 2) Look inside.
    print(f"Bodies:    {model.nbody}")
    print(f"Joints:    {model.njnt}  (7 arm joints + 2 finger joints + 3 free joints for the cubes)")
    print(f"Motors:    {model.nu}")
    print(f"Timestep:  {model.opt.timestep} s  ({1 / model.opt.timestep:.0f} physics steps per simulated second)")
    for i in range(model.nu):
        low, high = model.actuator(i).ctrlrange
        print(f"  motor {i}: {model.actuator(i).name:<10} command range [{low:+.2f}, {high:+.2f}]")

    # 3) Start from the "home" pose saved in the XML (keyframe number 0).
    mujoco.mj_resetDataKeyframe(model, data, 0)
    home = data.ctrl.copy()   # the 8 motor commands that hold the home pose

    # 4) The simulation loop: choose motor commands, step the physics, repeat.
    #    The 7 arm motors are position servos: ctrl[i] is the angle (radians) joint i should go to.
    #    Motor 7 drives both fingers: 255 = open, 0 = closed.
    out = open_output(model, data, ROOT / "videos" / "01_hello_robot.mp4", view=args.view)
    while data.time < 8.0:
        t = data.time
        data.ctrl[:] = home
        data.ctrl[0] = 0.8 * np.sin(2 * np.pi * t / 4)                 # joint 1: base swivels
        data.ctrl[3] = home[3] + 0.5 * np.sin(2 * np.pi * t / 2.5)     # joint 4: elbow bends
        data.ctrl[5] = home[5] + 0.4 * np.sin(2 * np.pi * t / 2)       # joint 6: wrist nods
        data.ctrl[7] = 255 if int(t) % 2 == 0 else 0                    # gripper: open, close, open...

        mujoco.mj_step(model, data)

        out.lines = [f"t = {t:4.1f} s",
                     f"joint 1  asked {data.ctrl[0]:+.2f}  actual {data.qpos[0]:+.2f} rad",
                     f"joint 4  asked {data.ctrl[3]:+.2f}  actual {data.qpos[3]:+.2f} rad",
                     f"gripper  {'open' if data.ctrl[7] else 'closed'}"]
        out()
    out.close()

    # ctrl is a request; qpos is reality. The motors need a moment to get there.
    print(f"\nAt the end, joint 1 was asked for {data.ctrl[0]:+.3f} rad and is at {data.qpos[0]:+.3f} rad.")


if __name__ == "__main__":
    main()
