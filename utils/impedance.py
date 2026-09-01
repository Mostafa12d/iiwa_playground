"""Cartesian impedance control with a nullspace posture task, for the iiwa.

Single source for the control law that collect.py and articulation_controller.py
both run. Works welded or free: the Jacobian spans all nv DOFs, but only the arm
DOFs are commanded.
"""

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class ControlOutput:
    """One step of the control law, plus the intermediates worth logging."""

    tau: np.ndarray           # commanded torque, clipped to ctrlrange [Nm]
    tau_unclipped: np.ndarray
    saturated: bool
    dx: np.ndarray            # position error, base frame [m]
    dtheta: np.ndarray        # orientation error as a rotation vector [rad]
    ee_pos: np.ndarray
    ee_quat: np.ndarray       # (w, x, y, z)
    ee_twist: np.ndarray      # [linear(3), angular(3)], base frame
    jac: np.ndarray           # (6, nv) site Jacobian


class CartesianImpedance:
    """Impedance controller for a 7-DOF iiwa in any scene containing one.

    :param kp_pos: translational stiffness [N/m], scalar or (3,)
    :param kp_ori: rotational stiffness [Nm/rad], scalar or (3,)
    :param kp_null: nullspace posture stiffness, (7,)
    :param kpos, kori: error gains on the twist, 0..1
    """

    def __init__(self, model, kp_pos, kp_ori, kp_null,
                 damping_ratio=1.0, kpos=0.95, kori=0.95,
                 integration_dt=1.0, gravity_compensation=True,
                 site_name="attachment_site", joint_names=None,
                 track_orientation=True):
        self.model = model
        self.site_id = model.site(site_name).id
        names = joint_names or [f"joint{i}" for i in range(1, 8)]
        jids = np.array([model.joint(n).id for n in names])
        self.dof_ids = model.jnt_dofadr[jids]
        self.qpos_ids = model.jnt_qposadr[jids]
        self.actuator_ids = np.array([model.actuator(n).id for n in names])

        self.Kp = np.concatenate([np.broadcast_to(kp_pos, 3),
                                  np.broadcast_to(kp_ori, 3)]).astype(float)
        self.Kd = damping_ratio * 2 * np.sqrt(self.Kp)
        self.Kp_null = np.asarray(kp_null, dtype=float)
        self.Kd_null = damping_ratio * 2 * np.sqrt(self.Kp_null)

        self.kpos = kpos
        self.kori = kori
        self.integration_dt = integration_dt
        self.gravity_compensation = gravity_compensation
        self.track_orientation = track_orientation
        self.q0 = None

        nv = model.nv
        self._jac = np.zeros((6, nv))
        self._twist = np.zeros(6)
        self._M_inv = np.zeros((nv, nv))
        self._eye_nv = np.eye(nv)
        self._q = [np.zeros(4) for _ in range(4)]
        self._dtheta = np.zeros(3)

    def capture_posture(self, data):
        """Freeze the current arm configuration as the nullspace reference."""
        self.q0 = data.qpos[self.qpos_ids].copy()
        return self.q0

    def compute(self, data, pos_des, quat_des):
        """Torque for the arm DOFs. Does not write ctrl or step."""
        if self.q0 is None:
            self.capture_posture(data)
        model = self.model
        jac, twist = self._jac, self._twist
        sq, sqc, eq, ee_quat = self._q
        dth = self._dtheta

        ee_pos = data.site(self.site_id).xpos.copy()
        dx = pos_des - ee_pos

        mujoco.mju_mat2Quat(sq, data.site(self.site_id).xmat)
        ee_quat[:] = sq
        mujoco.mju_negQuat(sqc, sq)
        mujoco.mju_mulQuat(eq, quat_des, sqc)
        mujoco.mju_quat2Vel(dth, eq, 1.0)

        twist[:3] = self.kpos * dx / self.integration_dt
        twist[3:] = (dth * (self.kori / self.integration_dt)
                     if self.track_orientation else 0.0)

        mujoco.mj_jacSite(model, data, jac[:3], jac[3:], self.site_id)
        ee_twist = jac @ data.qvel

        mujoco.mj_solveM(model, data, self._M_inv, self._eye_nv)
        Mx_inv = jac @ self._M_inv @ jac.T
        if abs(np.linalg.det(Mx_inv)) >= 1e-2:
            Mx = np.linalg.inv(Mx_inv)
        else:
            Mx = np.linalg.pinv(Mx_inv, rcond=1e-2)

        tau_full = jac.T @ Mx @ (self.Kp * twist - self.Kd * ee_twist)

        # Only the arm DOFs are actuated, so the posture term is written into
        # the arm block alone; other DOFs ride along in the Jacobian.
        Jbar = self._M_inv @ jac.T @ Mx
        ddq = np.zeros(model.nv)
        ddq[self.dof_ids] = (
            self.Kp_null * (self.q0 - data.qpos[self.qpos_ids])
            - self.Kd_null * data.qvel[self.dof_ids]
        )
        tau_full += (self._eye_nv - jac.T @ Jbar.T) @ ddq

        tau = tau_full[self.dof_ids]
        if self.gravity_compensation:
            tau = tau + data.qfrc_bias[self.dof_ids]

        unclipped = tau.copy()
        tau = np.clip(tau, *model.actuator_ctrlrange.T)
        return ControlOutput(
            tau=tau, tau_unclipped=unclipped,
            saturated=bool(np.any(np.abs(unclipped - tau) > 1e-9)),
            dx=dx, dtheta=dth.copy(), ee_pos=ee_pos, ee_quat=ee_quat.copy(),
            ee_twist=ee_twist, jac=jac,
        )

    def apply(self, data, out):
        """Write a ControlOutput's torque into data.ctrl."""
        data.ctrl[self.actuator_ids] = out.tau
