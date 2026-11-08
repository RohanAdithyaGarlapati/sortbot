"""Smooth gripper motions built on top of IK.

``Controller`` keeps a *target* pose for the gripper. Every physics step it runs one IK step
toward that target and sends the resulting joint angles to the robot's motors. To move
somewhere, we slide the target along a smooth path and let IK follow it.
"""

from __future__ import annotations

from typing import Callable, Optional

import numpy as np

from .ik import down_quat, ik_step
from .sim import GRIPPER_CLOSED, GRIPPER_OPEN, HOME_Q, Sim

TRAVEL_Z = 0.26                         # carry height: clears a full 3-cube tower
HOME_TCP = np.array([0.45, 0.0, 0.42])  # where the gripper rests between tasks

# The motors act like springs with shock absorbers. The shock absorber pushes back whenever a
# joint moves, so a moving gripper trails its target by about 0.1 s. The fix is to aim 0.1 s
# ahead along the planned path (in control terms, velocity feedforward).
LOOKAHEAD = 0.1


def min_jerk(s: float) -> float:
    """Progress curve from 0 to 1 that starts and stops gently (zero speed and acceleration)."""
    return 10 * s**3 - 15 * s**4 + 6 * s**5


class Controller:
    """Moves the gripper along smooth paths and opens or closes it."""

    def __init__(self, sim: Sim, on_step: Optional[Callable[[], None]] = None, speed: float = 0.4):
        self.sim = sim
        self.on_step = on_step      # called after every physics step, for example to record video
        self.speed = speed          # average speed of moves, meters per second
        self.target_pos = sim.tcp_pos()   # where the gripper should be right now
        self.target_yaw = 0.0
        self.aim_pos = self.target_pos.copy()  # what IK steers toward (slightly ahead when moving)
        self.aim_yaw = 0.0
        self.gripper = GRIPPER_OPEN
        self.status = ""            # what the robot is doing, shown as a caption in videos

    # ------------------------------------------------------------------ the control loop
    def tick(self) -> None:
        """One control cycle: IK toward the aim point, send motor commands, advance physics."""
        s = self.sim
        dq = ik_step(s.model, s.data, s.tcp_id, s.arm_qpos_ids, s.arm_dof_ids,
                     self.aim_pos, down_quat(self.aim_yaw), posture=HOME_Q)
        q_cmd = np.clip(s.arm_q() + dq, s.arm_limits[:, 0], s.arm_limits[:, 1])
        s.data.ctrl[:7] = q_cmd          # the 7 arm motors are position servos
        s.data.ctrl[7] = self.gripper    # 0 = close, 255 = open
        s.step()
        if self.on_step:
            self.on_step()

    def wait(self, seconds: float) -> None:
        for _ in range(int(round(seconds / self.sim.dt))):
            self.tick()

    def settle(self, tol: float = 0.0015, max_time: float = 0.4) -> None:
        """Hold still until the gripper is within ``tol`` meters of the target."""
        self.aim_pos, self.aim_yaw = self.target_pos.copy(), self.target_yaw
        deadline = self.sim.time + max_time
        while np.linalg.norm(self.target_pos - self.sim.tcp_pos()) > tol and self.sim.time < deadline:
            self.tick()

    # ------------------------------------------------------------------ primitives
    def move_to(self, pos, yaw: float | None = None, speed: float | None = None) -> None:
        """Glide the gripper to ``pos`` (and optionally turn it to ``yaw``) along a straight line."""
        start, goal = self.target_pos.copy(), np.asarray(pos, dtype=float)
        yaw0 = self.target_yaw
        yaw1 = yaw0 if yaw is None else float(yaw)
        duration = max(0.4, np.linalg.norm(goal - start) / (speed or self.speed), abs(yaw1 - yaw0) / 1.5)
        steps = max(1, int(round(duration / self.sim.dt)))
        lead = int(round(LOOKAHEAD / self.sim.dt))
        for i in range(1, steps + 1):
            p = min_jerk(i / steps)                       # where we should be now
            a = min_jerk(min(1.0, (i + lead) / steps))    # where we aim (0.1 s ahead)
            self.target_pos = start + p * (goal - start)
            self.target_yaw = yaw0 + p * (yaw1 - yaw0)
            self.aim_pos = start + a * (goal - start)
            self.aim_yaw = yaw0 + a * (yaw1 - yaw0)
            self.tick()
        self.settle()

    def open_gripper(self, settle: float = 0.35) -> None:
        self.gripper = GRIPPER_OPEN
        self.wait(settle)

    def close_gripper(self, settle: float = 0.5) -> None:
        self.gripper = GRIPPER_CLOSED
        self.wait(settle)

    def holding_something(self) -> bool:
        """Closed fingers stop around 5 cm apart when they grip a cube, and near 0 when they miss."""
        return self.sim.gripper_width() > 0.03

    def go_home(self) -> None:
        self.status = "going home"
        if self.target_pos[2] < TRAVEL_Z - 0.01:
            self.move_to([self.target_pos[0], self.target_pos[1], TRAVEL_Z])
        self.move_to(HOME_TCP, yaw=0.0)

    # ------------------------------------------------------------------ pick and place
    def grasp_yaw(self, color: str) -> float:
        """A cube looks the same every 90 degrees, so pick the equivalent angle closest to zero."""
        yaw = self.sim.cube_yaw(color)
        return float((yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4)

    def pick(self, color: str) -> bool:
        """Pick up a cube. Returns True if the gripper ends up holding it."""
        cube = self.sim.cube_pos(color)
        yaw = self.grasp_yaw(color)
        self.status = f"picking up the {color} cube"
        self.open_gripper(settle=0.0)
        if self.target_pos[2] < TRAVEL_Z - 0.01:                      # 0. rise to a safe height first
            self.move_to([self.target_pos[0], self.target_pos[1], TRAVEL_Z])
        self.move_to([cube[0], cube[1], TRAVEL_Z], yaw=yaw)          # 1. hover above it
        self.move_to([cube[0], cube[1], cube[2]], speed=0.25)       # 2. descend so fingertips surround it
        self.close_gripper()                                          # 3. squeeze
        self.move_to([cube[0], cube[1], TRAVEL_Z], speed=0.3)        # 4. lift
        return self.holding_something()

    def place(self, xyz, label: str = "") -> None:
        """Lower the held cube so its center ends up at ``xyz``, then let go."""
        x, y, z = xyz
        self.status = f"placing it {label}".strip()
        self.move_to([x, y, TRAVEL_Z])                       # 1. carry it over the target
        self.move_to([x, y, z + 0.004], speed=0.25)          # 2. lower it (a 4 mm drop is fine)
        self.open_gripper()                                  # 3. release
        self.move_to([x, y, TRAVEL_Z], speed=0.3)            # 4. back off upward
