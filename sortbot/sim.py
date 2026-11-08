"""Load the SortBot scene and give friendly names to the things inside it.

Everything MuJoCo knows about the world lives in two objects:

* ``model`` (MjModel): the fixed description. Link lengths, masses, joint limits, colors.
* ``data``  (MjData):  the changing state. Joint angles, velocities, contact forces, time.

The ``Sim`` class below just wraps those two so the lessons can say ``sim.cube_pos("red")``
instead of digging through arrays by index.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
SCENE_PATH = ROOT / "robot" / "scene.xml"

COLORS = ("red", "green", "blue")
ARM_JOINTS = [f"joint{i}" for i in range(1, 8)]

# The "home" pose: 7 joint angles in radians (same as the keyframe in panda.xml).
HOME_Q = np.array([0.0, 0.0, 0.0, -1.57079, 0.0, 1.57079, -0.7853])

# Gripper command range comes from the model: 255 = fully open (8 cm), 0 = squeeze shut.
GRIPPER_OPEN = 255.0
GRIPPER_CLOSED = 0.0

CUBE_HALF = 0.025  # cubes are 5 cm on each side


class Sim:
    """A loaded SortBot world: the robot, three cubes and three pads."""

    def __init__(self, scene_path: Path | str = SCENE_PATH):
        self.model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.data = mujoco.MjData(self.model)
        self.dt = self.model.opt.timestep  # 0.002 s, so 500 physics steps per second

        # Look up indices once, by name.
        self.tcp_id = self.model.site("tcp").id
        self.arm_qpos_ids = np.array([self.model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.arm_dof_ids = np.array([self.model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.arm_limits = np.array([self.model.joint(j).range for j in ARM_JOINTS])  # shape (7, 2)
        self.finger_qpos_ids = [self.model.joint(j).qposadr[0] for j in ("finger_joint1", "finger_joint2")]
        self.start_positions = {c: self.data.body(f"{c}_cube").xpos.copy() for c in COLORS}
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self, cube_xy: dict[str, tuple[float, float]] | None = None) -> None:
        """Put the robot in its home pose and the cubes on the floor.

        ``cube_xy`` optionally moves cubes, for example ``{"red": (0.5, 0.1)}``.
        """
        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)  # keyframe 0 is "home"
        if cube_xy:
            for color, (x, y) in cube_xy.items():
                self.set_cube_pose(color, (x, y, CUBE_HALF))
        mujoco.mj_forward(self.model, self.data)  # recompute positions without advancing time

    def set_cube_pose(self, color: str, pos, yaw: float = 0.0) -> None:
        """Teleport a cube. A free joint stores 7 numbers: x, y, z and a quaternion (w, x, y, z)."""
        joint = self.model.joint(f"{color}_cube")
        adr = joint.qposadr[0]
        self.data.qpos[adr:adr + 3] = pos
        self.data.qpos[adr + 3:adr + 7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
        self.data.qvel[joint.dofadr[0]:joint.dofadr[0] + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)  # update positions so queries see the new pose

    def random_layout(self, seed: int) -> dict[str, tuple[float, float]]:
        """Pick random, non-overlapping cube spots in front of the robot, on its right side."""
        rng = np.random.default_rng(seed)
        spots: list[np.ndarray] = []
        while len(spots) < len(COLORS):
            p = np.array([rng.uniform(0.38, 0.66), rng.uniform(-0.26, 0.0)])
            if all(np.linalg.norm(p - s) > 0.1 for s in spots):
                spots.append(p)
        return {c: (float(p[0]), float(p[1])) for c, p in zip(COLORS, spots)}

    # ------------------------------------------------------------------ stepping
    def step(self) -> None:
        """Advance physics by one timestep (2 ms)."""
        mujoco.mj_step(self.model, self.data)

    @property
    def time(self) -> float:
        return float(self.data.time)

    # ------------------------------------------------------------------ queries
    def arm_q(self) -> np.ndarray:
        """Current 7 joint angles (radians)."""
        return self.data.qpos[self.arm_qpos_ids].copy()

    def tcp_pos(self) -> np.ndarray:
        """Where the fingertips are, in world coordinates (meters)."""
        return self.data.site_xpos[self.tcp_id].copy()

    def tcp_quat(self) -> np.ndarray:
        """Orientation of the gripper as a quaternion (w, x, y, z)."""
        quat = np.empty(4)
        mujoco.mju_mat2Quat(quat, self.data.site_xmat[self.tcp_id])
        return quat

    def gripper_width(self) -> float:
        """Distance between the fingers in meters (0 to 0.08)."""
        return float(sum(self.data.qpos[i] for i in self.finger_qpos_ids))

    def cube_pos(self, color: str) -> np.ndarray:
        return self.data.body(f"{color}_cube").xpos.copy()

    def cube_yaw(self, color: str) -> float:
        """Rotation of a cube about the vertical axis, in radians."""
        R = self.data.body(f"{color}_cube").xmat.reshape(3, 3)
        return float(np.arctan2(R[1, 0], R[0, 0]))

    def pad_pos(self, color: str) -> np.ndarray:
        return self.data.geom(f"{color}_pad").xpos.copy()

    def cube_on_pad(self, color: str, pad_color: str | None = None, tol: float = 0.03) -> bool:
        """True if the cube rests on the floor of the given pad (default: its own color)."""
        pad = self.pad_pos(pad_color or color)
        cube = self.cube_pos(color)
        return bool(np.linalg.norm(cube[:2] - pad[:2]) < tol and cube[2] < CUBE_HALF + 0.01)
