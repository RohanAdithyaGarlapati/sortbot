"""Lesson 3: Move the hand anywhere. (inverse kinematics)

Inverse kinematics (IK) is the reverse of lesson 2: given where the gripper SHOULD be,
find joint angles that put it there. There is no neat formula for a 7-joint arm, so we
solve it a little at a time, 500 times per second:

    1. error = target - where the gripper is now          (6 numbers: position + rotation)
    2. J     = the Jacobian: how each joint moves the gripper (6 x 7 matrix, from MuJoCo)
    3. dq    = small joint change that best reduces the error:
               dq = J^T (J J^T + lambda^2 I)^-1 error      <- damped least squares
    4. send q + dq to the motors, step physics, repeat

Step 3 is ridge regression: J is the feature matrix, the error is the target, dq are the
weights, and lambda is the ridge penalty that keeps dq sensible when the arm is stretched out.

The gripper traces a figure-eight in the air while pointing straight down.

Run it:
    python lessons/03_inverse_kinematics.py                  # records videos/03_inverse_kinematics.mp4
    python lessons/03_inverse_kinematics.py --view
Experiments:
    --no-lookahead          aim at the target itself instead of 0.1 s ahead: watch the error grow
    --stretch --damping 0   send the target out of reach with no ridge penalty: watch the arm thrash
"""

import argparse
from pathlib import Path

import mujoco
import numpy as np

from sortbot.ik import down_quat, orientation_error
from sortbot.video import open_output

ROOT = Path(__file__).resolve().parent.parent
HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])


def ik_step(model, data, site_id, target_pos, target_quat, damping):
    """One damped-least-squares step for the 7 arm joints. Returns dq (radians)."""
    # 1) error: 3 numbers of position (m) and 3 of rotation (rad)
    current_quat = np.empty(4)
    mujoco.mju_mat2Quat(current_quat, data.site_xmat[site_id])
    error = np.concatenate([target_pos - data.site_xpos[site_id],
                            orientation_error(target_quat, current_quat)])

    # 2) Jacobian: rows = the 6 error directions, columns = the 7 arm joints
    jac_pos, jac_rot = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)
    J = np.vstack([jac_pos, jac_rot])[:, :7]

    # 3) damped least squares (ridge regression)
    dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(6), error)

    # Bonus: 7 joints but only 6 numbers to match leaves 1 spare. Use it to drift toward the home
    # posture, projected into the "null space" so it never moves the gripper itself.
    null_space = np.eye(7) - np.linalg.pinv(J) @ J
    dq += null_space @ (0.05 * (HOME_Q - data.qpos[:7]))
    return np.clip(dq, -0.2, 0.2)


def figure_eight(t, stretch=False):
    """Target position at time t: a figure-eight 30 cm above the floor."""
    s = 2 * np.pi * t / 6.0                      # one loop every 6 seconds
    x = 0.5 + 0.12 * np.sin(2 * s)
    y = 0.22 * np.sin(s)
    if stretch:
        x += 0.5 * max(0.0, np.sin(s)) ** 8      # an excursion to x ~ 1.1 m, out of reach
    return np.array([x, y, 0.30])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--view", action="store_true", help="open a live window instead of recording")
    parser.add_argument("--damping", type=float, default=0.02, help="lambda, the ridge penalty")
    parser.add_argument("--no-lookahead", action="store_true", help="aim at the target, not 0.1 s ahead")
    parser.add_argument("--stretch", action="store_true", help="push the target out of reach")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(ROOT / "robot" / "scene.xml"))
    data = mujoco.MjData(model)
    tcp = model.site("tcp").id
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)               # compute positions for the home pose before we read them

    target_quat = down_quat(0.0)                 # fingers pointing at the floor
    lookahead = 0.0 if args.no_lookahead else 0.1
    out = open_output(model, data, ROOT / "videos" / "03_inverse_kinematics.mp4", view=args.view)

    # Glide from home to the start of the path first (2 s), then trace two loops (12 s).
    start = data.site_xpos[tcp].copy()
    errors, trail_target, trail_actual, next_dot, fastest = [], [], [], 0.0, 0.0
    while data.time < 14.0:
        t = data.time
        if t < 2.0:
            a = min(1.0, (t + lookahead) / 2.0)
            target, aim = start + (t / 2.0) * (figure_eight(0) - start), start + a * (figure_eight(0) - start)
        else:
            target, aim = figure_eight(t - 2.0, args.stretch), figure_eight(t - 2.0 + lookahead, args.stretch)

        try:
            dq = ik_step(model, data, tcp, aim, target_quat, args.damping)
        except np.linalg.LinAlgError:   # only possible with --damping 0, exactly at a singularity
            print(f"Stopped at t = {t:.2f} s: J J^T could not be inverted. Use a damping above 0.")
            break
        data.ctrl[:7] = data.qpos[:7] + dq
        mujoco.mj_step(model, data)

        actual = data.site_xpos[tcp]
        if t > 2.0:
            errors.append(np.linalg.norm(target - actual))
            fastest = max(fastest, np.abs(data.qvel[:7]).max())
        if t >= next_dot:
            trail_target.append((target.copy(), (1, 1, 1, 0.35), 0.004))
            trail_actual.append((actual.copy(), (1.0, 0.55, 0.1, 1.0), 0.006))
            next_dot += 0.04
        out.markers = trail_target[-300:] + trail_actual[-300:]
        out.lines = [f"damping (lambda) = {args.damping}   lookahead = {lookahead} s",
                     f"tracking error = {1000 * np.linalg.norm(target - actual):5.1f} mm"]
        out()
    out.close()

    if errors:
        errors = 1000 * np.array(errors)
        print(f"Tracking error: mean {errors.mean():.1f} mm, worst {errors.max():.1f} mm")
        print(f"Fastest joint speed: {fastest:.2f} rad/s  (the real Panda tops out around 2.2 to 2.6)")


if __name__ == "__main__":
    main()
