"""Inverse kinematics (IK): "which joint angles put the gripper here?"

We solve it a little at a time (differential IK). Every physics step:

1. Measure the error: where the gripper should be minus where it is (plus an orientation error).
2. Ask MuJoCo for the Jacobian J. Row i, column j says how much error component i
   changes when joint j moves a tiny bit.
3. Solve for small joint changes dq that shrink the error. We use damped least squares:

       dq = J^T (J J^T + lambda^2 I)^-1 * error

   which minimizes  ||J dq - error||^2 + lambda^2 ||dq||^2.
   That is exactly ridge regression, with J as the feature matrix and dq as the weights.
   The damping term keeps dq small and well behaved when J is badly conditioned (a "singularity").

This file is the same code you build step by step in lessons/03_inverse_kinematics.py.
"""

from __future__ import annotations

import mujoco
import numpy as np

# Gripper pointing straight down, fingers opening along the world x axis (the home orientation).
R_DOWN = np.array([[0.0, 1.0, 0.0],
                   [1.0, 0.0, 0.0],
                   [0.0, 0.0, -1.0]])


def down_quat(yaw: float = 0.0) -> np.ndarray:
    """Quaternion for "gripper points down, rotated by yaw about the vertical axis"."""
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    quat = np.empty(4)
    mujoco.mju_mat2Quat(quat, (Rz @ R_DOWN).flatten())
    return quat


def orientation_error(target_quat: np.ndarray, current_quat: np.ndarray) -> np.ndarray:
    """Rotation vector (axis * angle, world frame) that turns the current orientation into the target."""
    current_conj = np.empty(4)
    mujoco.mju_negQuat(current_conj, current_quat)
    error_quat = np.empty(4)
    mujoco.mju_mulQuat(error_quat, target_quat, current_conj)
    error_vec = np.empty(3)
    mujoco.mju_quat2Vel(error_vec, error_quat, 1.0)
    return error_vec


def ik_step(model, data, site_id, qpos_ids, dof_ids, target_pos, target_quat=None,
            damping=0.02, posture=None, posture_gain=0.05, max_dq=0.2):
    """Return a small joint-angle change dq that moves ``site_id`` toward the target pose.

    Args:
        target_pos:  desired (x, y, z) of the site, meters.
        target_quat: desired orientation (w, x, y, z), or None to only control position.
        damping:     lambda in damped least squares (the ridge penalty).
        posture:     optional preferred joint angles. The arm has 7 joints but the task only
                     needs 6, so there is one spare degree of freedom. We use it to stay close to
                     a comfortable posture without disturbing the gripper (a "null space" task).
        max_dq:      safety clamp on the change per step, radians.
    """
    # 1) Error vector: position (meters) and, optionally, orientation (radians).
    pos_err = np.asarray(target_pos) - data.site_xpos[site_id]
    jac_pos = np.zeros((3, model.nv))
    jac_rot = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jac_pos, jac_rot, site_id)  # 2) the Jacobian

    if target_quat is None:
        error, J = pos_err, jac_pos[:, dof_ids]
    else:
        current_quat = np.empty(4)
        mujoco.mju_mat2Quat(current_quat, data.site_xmat[site_id])
        rot_err = orientation_error(target_quat, current_quat)
        error = np.concatenate([pos_err, rot_err])
        J = np.vstack([jac_pos, jac_rot])[:, dof_ids]

    # 3) Damped least squares, the ridge regression solution.
    JJt = J @ J.T + damping ** 2 * np.eye(J.shape[0])
    dq = J.T @ np.linalg.solve(JJt, error)

    # Optional: drift toward a preferred posture using only the spare (null space) motion.
    if posture is not None:
        J_pinv = J.T @ np.linalg.inv(JJt)
        null_space = np.eye(len(dof_ids)) - J_pinv @ J
        dq += null_space @ (posture_gain * (posture - data.qpos[qpos_ids]))

    return np.clip(dq, -max_dq, max_dq)
