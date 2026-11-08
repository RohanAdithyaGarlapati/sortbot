"""Lesson 2: Where is the hand? (forward kinematics)

Forward kinematics (FK) answers one question: given the 7 joint angles, where is the gripper?

Each link is attached to the previous one with a fixed offset (from the XML) and a joint that
rotates about its own z axis. Multiply those transforms down the chain and you get the
gripper's pose. We do it by hand with numpy, check the answer against MuJoCo, then move two
joints and draw the path the fingertips trace through the air.

Run it:
    python lessons/02_forward_kinematics.py           # records videos/02_forward_kinematics.mp4
    python lessons/02_forward_kinematics.py --view
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from sortbot.video import make_camera, open_output

ROOT = Path(__file__).resolve().parent.parent
CHAIN = ["link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand"]


def quat_to_matrix(q):
    """Rotation matrix from a unit quaternion (w, x, y, z)."""
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def transform(rotation=np.eye(3), translation=np.zeros(3)):
    """4x4 homogeneous transform: rotate, then shift."""
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = translation
    return T


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def forward_kinematics(model, q):
    """Fingertip (tcp) position for the 7 joint angles in q, computed link by link."""
    T = np.eye(4)                       # start at the world origin
    joint_index = 0
    for name in CHAIN:
        body = model.body(name)
        T = T @ transform(quat_to_matrix(body.quat), body.pos)   # fixed offset from the parent link
        if body.jntnum[0] > 0 and joint_index < 7:               # links 1 to 7 each have a hinge
            T = T @ transform(rot_z(q[joint_index]))              # rotate about the local z axis
            joint_index += 1
    T = T @ transform(translation=model.site("tcp").pos)          # fingertips: 10.34 cm past the hand
    return T[:3, 3]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--view", action="store_true", help="open a live window instead of recording")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(ROOT / "robot" / "scene.xml"))
    data = mujoco.MjData(model)
    tcp = model.site("tcp").id

    # --- Check: our 20 lines of numpy against the physics engine, for a few random poses.
    rng = np.random.default_rng(0)
    print("Forward kinematics check (meters):")
    for _ in range(3):
        q = rng.uniform(-1.0, 1.0, size=7)
        q[3] = rng.uniform(-2.5, -0.5)          # joint 4 only bends one way
        data.qpos[:7] = q
        mujoco.mj_kinematics(model, data)       # MuJoCo's own FK
        mine, theirs = forward_kinematics(model, q), data.site_xpos[tcp]
        print(f"  mine {np.round(mine, 4)}   mujoco {np.round(theirs, 4)}   difference {np.linalg.norm(mine - theirs):.1e}")

    # --- Watch it: sweep joint 1 (base) and joint 4 (elbow), and trace the fingertips.
    mujoco.mj_resetDataKeyframe(model, data, 0)
    home = data.ctrl.copy()
    wide = make_camera(lookat=(0.3, 0.0, 0.4), distance=2.0)      # zoomed out: the arm reaches high
    out = open_output(model, data, ROOT / "videos" / "02_forward_kinematics.mp4", view=args.view, camera=wide)
    trail, next_dot = [], 0.0
    while data.time < 10.0:
        t = data.time
        data.ctrl[:] = home
        data.ctrl[0] = 1.0 * np.sin(2 * np.pi * t / 5)                        # base: left-right
        data.ctrl[3] = home[3] + 0.6 * np.sin(2 * np.pi * t / 2.5) - 0.2      # elbow: in-out
        mujoco.mj_step(model, data)

        q = data.qpos[:7]
        pos = forward_kinematics(model, q)
        if t >= next_dot:                                   # drop a breadcrumb every 40 ms
            trail.append((pos.copy(), (1.0, 0.55, 0.1, 1.0), 0.006))
            next_dot += 0.04
        out.markers = trail[-250:]
        out.lines = ["joints (rad): " + " ".join(f"{a:+.2f}" for a in q),
                     f"fingertips (m): x={pos[0]:.3f}  y={pos[1]:.3f}  z={pos[2]:.3f}"]
        out()
    out.close()


if __name__ == "__main__":
    main()
