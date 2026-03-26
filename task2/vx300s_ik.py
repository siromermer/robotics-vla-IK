#!/usr/bin/env python3
"""From-scratch FK and damped least-squares IK for the Trossen VX300s (6 DOF).

Purpose
    Match the MuJoCo Menagerie ``vx300s.xml`` kinematic tree: multiply link
    offsets and joint rotations, place the pinch site, build a 6x6 geometric
    Jacobian, and iterate small joint updates to reach a target EE pose.

Who calls this
    ``forward_kinematics`` and ``solve_ik`` / ``solve_ik_trajectory`` from
    ``task2.teleop_main.build_trajectory`` (home pose) and ``main`` (full traj).

Joint vector ``q``
    Shape (6,) radians, order:
    waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate.

End effector
    Position (3,) and rotation (3, 3) are for the pinch point (offset in link
    frame from ``EE_OFFSET``).

Constants below (``LINK_OFFSETS``, ``JOINT_AXES``, limits) are taken from the MJCF.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

JOINT_NAMES = [
    "waist", "shoulder", "elbow",
    "forearm_roll", "wrist_angle", "wrist_rotate",
]

NUM_JOINTS = 6

LINK_OFFSETS = np.array([
    [0.0,      0.0, 0.079],
    [0.0,      0.0, 0.04805],
    [0.05955,  0.0, 0.3],
    [0.2,      0.0, 0.0],
    [0.1,      0.0, 0.0],
    [0.069744, 0.0, 0.0],
])

JOINT_AXES = np.array([
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 0],
    [1, 0, 0],
    [0, 1, 0],
    [1, 0, 0],
])

JOINT_LIMITS = np.array([
    [-3.14158,  3.14158],
    [-1.85005,  1.25664],
    [-1.76278,  1.6057],
    [-3.14158,  3.14158],
    [-1.8675,   2.23402],
    [-3.14158,  3.14158],
])

EE_OFFSET = np.array([0.1, 0.0, 0.0])

HOME_QPOS = np.array([0.0, -0.96, 1.16, 0.0, -0.3, 0.0])


def rot_x(a: float) -> np.ndarray:
    """Elementary rotation about X; output (3, 3). Used by teleop orientation deltas."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a: float) -> np.ndarray:
    """Elementary rotation about Y; output (3, 3)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a: float) -> np.ndarray:
    """Elementary rotation about Z; output (3, 3)."""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


_ROT_FN = [rot_z, rot_y, rot_y, rot_x, rot_y, rot_x]


def _make_transform(offset: np.ndarray, R: np.ndarray) -> np.ndarray:
    """Rigid transform 4x4 from translation ``offset`` (3,) and rotation R (3,3)."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = offset
    return T


def rotation_matrix_to_axis_angle(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Return unit axis (3,) and angle in radians for rotation matrix R (3,3)."""
    cos_a = np.clip((np.trace(R) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos_a)

    if angle < 1e-8:
        return np.array([0.0, 0.0, 1.0]), 0.0

    if np.pi - angle < 1e-6:
        eigvals, eigvecs = np.linalg.eig(R)
        idx = np.argmin(np.abs(eigvals - 1.0))
        axis = np.real(eigvecs[:, idx])
        return axis / np.linalg.norm(axis), angle

    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    return axis / (2.0 * np.sin(angle)), angle


class FKResult(NamedTuple):
    """Outputs of ``forward_kinematics``."""

    position: np.ndarray
    rotation: np.ndarray
    transform: np.ndarray
    joint_positions: list
    joint_axes_world: list


def forward_kinematics(q: np.ndarray) -> FKResult:
    """Forward kinematics to the pinch frame.

    Input
        ``q``: (6,) joint angles rad.

    Output
        ``FKResult`` with EE position (3,), rotation (3,3), full 4x4 transform,
        and per-joint origin positions and world axes for Jacobian rows.

    Called by
        ``compute_jacobian``, ``solve_ik``, ``teleop_main.build_trajectory`` (home).
    """
    T = np.eye(4)  # running transform from base to current link frame
    joint_positions: list[np.ndarray] = []
    joint_axes_world: list[np.ndarray] = []

    for i in range(NUM_JOINTS):
        # Apply the MJCF link offset, then the revolute joint rotation.
        T = T @ _make_transform(LINK_OFFSETS[i], np.eye(3))

        joint_positions.append(T[:3, 3].copy())
        joint_axes_world.append((T[:3, :3] @ JOINT_AXES[i]).copy())

        T = T @ _make_transform(np.zeros(3), _ROT_FN[i](q[i]))  # joint rotation about axis i

    T = T @ _make_transform(EE_OFFSET, np.eye(3))  # pinch site offset from final link

    return FKResult(
        position=T[:3, 3].copy(),
        rotation=T[:3, :3].copy(),
        transform=T.copy(),
        joint_positions=joint_positions,
        joint_axes_world=joint_axes_world,
    )


def compute_jacobian(q: np.ndarray) -> np.ndarray:
    """Spatial geometric Jacobian J (6, 6) in world frame.

    For revolute joint i: linear column is z_i cross (p_ee minus p_i), angular
    column is z_i. Rows 0 to 2 are linear, 3 to 5 are angular.

    Input ``q``: (6,). Output ``J``: (6, 6).

    Called by
        ``solve_ik`` each iteration.
    """
    fk = forward_kinematics(q)  # provides joint origins and axes in world frame
    p_ee = fk.position
    J = np.zeros((6, NUM_JOINTS))
    for i in range(NUM_JOINTS):
        z = fk.joint_axes_world[i]
        J[:3, i] = np.cross(z, p_ee - fk.joint_positions[i])
        J[3:, i] = z
    return J


def orientation_error(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """Rotation vector (3,) that rotates ``R_current`` toward ``R_target``."""
    R_err = R_target @ R_current.T
    axis, angle = rotation_matrix_to_axis_angle(R_err)
    return axis * angle


def solve_ik(
    target_pos: np.ndarray,
    target_rot: np.ndarray | None = None,
    q_init: np.ndarray | None = None,
    *,
    max_iter: int = 200,
    pos_tol: float = 1e-4,
    orient_tol: float = 5e-3,
    damping: float = 0.05,
    orient_weight: float = 0.3,
) -> tuple[np.ndarray, float, float, bool]:
    """One pose: damped least squares on stacked position and weighted orientation error.

    Inputs
        ``target_pos``: (3,) metres.
        ``target_rot``: (3, 3) or None for position-only IK.
        ``q_init``: (6,) warm start; default ``HOME_QPOS``.

    Outputs
        Tuple ``(q, pos_err_norm, orient_err_norm, converged)`` with ``q`` (6,).

    Called by
        ``solve_ik_trajectory`` for each frame.
    """
    q = q_init.copy() if q_init is not None else HOME_QPOS.copy()  # warm start for faster convergence
    pos_only = target_rot is None
    dim = 3 if pos_only else 6

    for _ in range(max_iter):
        fk = forward_kinematics(q)  # current EE pose under current q

        e_pos = target_pos - fk.position
        e_pos_n = np.linalg.norm(e_pos)

        if pos_only:
            if e_pos_n < pos_tol:
                return q, e_pos_n, 0.0, True
            e = e_pos
            J = compute_jacobian(q)[:3]
        else:
            e_ori = orientation_error(target_rot, fk.rotation)  # rotation-vector error (small-angle)
            e_ori_n = np.linalg.norm(e_ori)
            if e_pos_n < pos_tol and e_ori_n < orient_tol:
                return q, e_pos_n, e_ori_n, True
            e = np.concatenate([e_pos, orient_weight * e_ori])
            J = compute_jacobian(q).copy()
            J[3:] *= orient_weight

        JJt = J @ J.T  # damped least squares (DLS): stabilize near singularities
        dq = J.T @ np.linalg.solve(JJt + damping**2 * np.eye(dim), e)
        q = np.clip(q + dq, JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1])

    fk_f = forward_kinematics(q)
    pe = np.linalg.norm(target_pos - fk_f.position)
    oe = 0.0 if pos_only else np.linalg.norm(orientation_error(target_rot, fk_f.rotation))
    return q, pe, oe, False


def solve_ik_trajectory(
    positions: np.ndarray,
    orientations: np.ndarray | None = None,
    q_init: np.ndarray | None = None,
    **ik_kwargs,
) -> np.ndarray:
    """IK for N poses; each frame warm-starts from previous ``q``.

    Inputs
        ``positions``: (N, 3).
        ``orientations``: (N, 3, 3) or None.
        ``q_init``: (6,) seed for frame 0.

    Output
        ``q_traj`` with shape (N, 6).

    Called by
        ``task2.teleop_main.main``.
    """
    N = len(positions)
    q_traj = np.zeros((N, NUM_JOINTS))
    q = q_init.copy() if q_init is not None else HOME_QPOS.copy()

    for i in range(N):
        rot = orientations[i] if orientations is not None else None
        q, pe, oe, ok = solve_ik(positions[i], rot, q, **ik_kwargs)
        q_traj[i] = q
        if i % 30 == 0:
            tag = "OK" if ok else "  "
            msg = f"  IK {i:4d}/{N}: pos_err={pe:.5f}"
            if rot is not None:
                msg += f"  ori_err={oe:.5f}"
            msg += f"  [{tag}]"
            print(msg)

    return q_traj


def _self_test() -> None:
    """CLI sanity check: FK at home and optional MuJoCo pinch site comparison."""
    fk = forward_kinematics(HOME_QPOS)
    print("FK at HOME_QPOS:")
    print(f"  EE position : {fk.position}")
    print(f"  EE rotation :\n{fk.rotation}\n")

    try:
        import mujoco
        from pathlib import Path

        xml = Path(__file__).resolve().parent / "assets" / "trossen_vx300s" / "scene.xml"
        model = mujoco.MjModel.from_xml_path(str(xml))
        data = mujoco.MjData(model)

        if model.nkey > 0:
            mujoco.mj_resetDataKeyframe(model, data, 0)
        else:
            data.qpos[:NUM_JOINTS] = HOME_QPOS
        mujoco.mj_forward(model, data)

        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "pinch")
        mj_pos = data.site_xpos[site_id].copy()
        mj_rot = data.site_xmat[site_id].reshape(3, 3).copy()

        pos_err = np.linalg.norm(fk.position - mj_pos)
        print(f"MuJoCo pinch position : {mj_pos}")
        print(f"Position error vs MuJoCo: {pos_err:.6f} m")
        print(f"MuJoCo pinch rotation :\n{mj_rot}")
    except Exception as exc:
        print(f"(MuJoCo comparison skipped: {exc})")


if __name__ == "__main__":
    _self_test()
