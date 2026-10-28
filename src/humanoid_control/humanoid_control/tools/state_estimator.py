"""
Measured balance state for humanoid_walk (base ground truth + joint states + ankle FT).

The estimator keeps its own Pinocchio configuration and data so that it never disturbs the
IK solver. All outputs are expressed in the gait plane frame of the walk (x forward, y left),
anchored at the measured base pose when the walk started (`set_walk_reference_frame`).
"""

from dataclasses import dataclass
import threading
from typing import Dict, Optional

import numpy as np
import pinocchio as pin

from humanoid_control.tools.balance import BalanceEstimate
from humanoid_control.tools.support_polygon import convex_hull

# Minimum normal force [N] for a foot to count as loaded (about 10% of the robot weight).
MIN_CONTACT_FORCE = 30.0


@dataclass
class FootWrench:
    """Ankle joint wrench (force [N], torque [Nm]) in the foot frame, at the ankle."""

    force: np.ndarray
    torque: np.ndarray


def axial_to_plane(frame, vec) -> np.ndarray:
    """
    Rotation vector / angular velocity -> right-handed [about forward, about left].

    The plane axes need not form a right-handed basis in base coordinates (this robot's
    left foot is at base +x while it walks along +y), so axial vectors get the handedness
    sign; a positive rotation about forward then always tilts the body to the right
    (towards -y) and a positive rotation about left tilts it forward.
    """
    handed = np.sign(np.cross(frame.forward, frame.left) @ frame.up)
    return handed * frame.to_plane(np.asarray(vec, float))[:2]


def cop_from_ankle_wrench(force, torque, ankle_to_sole: float) -> Optional[np.ndarray]:
    """
    Centre of pressure (foot frame x, y) from the ankle wrench; None when unloaded.

    The ratio is independent of the wrench sign convention (joint wrench on the foot or
    ground reaction), so the joint feedback can be used directly. Foot inertia is ignored.
    """
    fz = float(force[2])
    if abs(fz) < MIN_CONTACT_FORCE:
        return None
    d = float(ankle_to_sole)
    px = (-float(torque[1]) - float(force[0]) * d) / fz
    py = (float(torque[0]) - float(force[1]) * d) / fz
    return np.array([px, py])


class BalanceStateEstimator:
    """Measured CoM, DCM, ZMP, support polygon and tilt in the gait plane frame."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = float(alpha)
        self.base_world: Optional[pin.SE3] = None
        self.base_vel_world = np.zeros(3)
        self.base_omega_world = np.zeros(3)
        self.base_time: Optional[float] = None
        self.ref_world: Optional[pin.SE3] = None
        self.joint_state: Dict[str, tuple] = {}
        self.wrenches: Dict[str, FootWrench] = {}
        self._wrench_sums: Dict[str, list] = {}
        self._wrench_lock = threading.Lock()
        self._model = None
        self._data = None
        self._q = None
        self._v = None
        self._com_vel: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ inputs
    def set_base(self, base_world: pin.SE3, stamp: float, lin_vel=None, ang_vel=None) -> None:
        """Set the base pose in the world, optionally with its world-frame velocities."""
        self.base_world = base_world
        self.base_time = float(stamp)
        if lin_vel is not None:
            self.base_vel_world = np.asarray(lin_vel, float)
        if ang_vel is not None:
            self.base_omega_world = np.asarray(ang_vel, float)

    def set_walk_reference_frame(self) -> bool:
        """Anchor the plane frame at the current measured base pose; False if unknown."""
        self._com_vel = None
        if self.base_world is None:
            self.ref_world = None
            return False
        self.ref_world = self.base_world.copy()
        return True

    def set_joint_state(self, names, positions, velocities=None, efforts=None) -> None:
        velocities = velocities if velocities is not None else []
        efforts = efforts if efforts is not None else []
        for i, name in enumerate(names):
            pos = float(positions[i]) if i < len(positions) else 0.0
            vel = float(velocities[i]) if i < len(velocities) else 0.0
            eff = float(efforts[i]) if i < len(efforts) else 0.0
            self.joint_state[name] = (pos, vel, eff)

    def set_wrench(self, side: str, force, torque) -> None:
        """
        Add a wrench sample; estimate() uses the mean of the samples since its last call.

        The sensors publish much faster than the control loop, and the stiff contact makes
        single samples noisy, so averaging over the loop period avoids aliasing.
        """
        with self._wrench_lock:
            acc = self._wrench_sums.setdefault(side, [np.zeros(3), np.zeros(3), 0])
            acc[0] = acc[0] + np.asarray(force, float)
            acc[1] = acc[1] + np.asarray(torque, float)
            acc[2] += 1

    def _consume_wrenches(self) -> None:
        with self._wrench_lock:
            for side, (force, torque, n) in self._wrench_sums.items():
                if n > 0:
                    self.wrenches[side] = FootWrench(force / n, torque / n)
            self._wrench_sums.clear()

    # ------------------------------------------------------------------ helpers
    def _bind(self, kin) -> None:
        if self._model is not kin.model:
            self._model = kin.model
            self._data = kin.model.createData()
            self._q = pin.neutral(kin.model)
            self._v = np.zeros(kin.model.nv)

    def _plane_rotvec(self, frame, rotation_world: np.ndarray) -> np.ndarray:
        """Rotation (relative to the walk-start base) as [about forward, about left]."""
        return axial_to_plane(frame, pin.log3(self.ref_world.rotation.T @ rotation_world))

    # ------------------------------------------------------------------ estimate
    def estimate(self, kin, frame, ground_height: float, contact_tol: float, omega: float,
                 ankle_to_sole: float) -> Optional[BalanceEstimate]:
        """Return the measured balance state, or None before the base pose is known."""
        if self.base_world is None or self.ref_world is None:
            return None
        self._bind(kin)
        self._consume_wrenches()
        model, data, q, v = self._model, self._data, self._q, self._v

        rot = self.base_world.rotation
        q[:7] = pin.SE3ToXYZQUAT(self.base_world)
        v[:3] = rot.T @ self.base_vel_world
        v[3:6] = rot.T @ self.base_omega_world
        for name, (pos, vel, _) in self.joint_state.items():
            jid = kin.joint_name_to_id.get(name)
            if jid is None or model.joints[jid].nq != 1:
                continue
            q[model.idx_qs[jid]] = pos
            v[model.idx_vs[jid]] = vel
        com_world = np.array(pin.centerOfMass(model, data, q, v))
        com_vel_world = np.array(data.vcom[0])
        pin.updateFramePlacements(model, data)

        if self._com_vel is None or self.alpha >= 1.0:
            self._com_vel = com_vel_world
        else:
            self._com_vel = (1.0 - self.alpha) * self._com_vel + self.alpha * com_vel_world

        ref_inv = self.ref_world.inverse()
        com = frame.to_plane(ref_inv.act(com_world))
        com_vel = frame.to_plane(self.ref_world.rotation.T @ self._com_vel)
        dcm = com[:2] + com_vel[:2] / max(omega, 1e-6)

        support_points = []
        foot_tilt = {}
        foot_force = {}
        zmp_sum = np.zeros(2)
        fz_sum = 0.0
        for side, fid in kin.foot_frame_ids.items():
            foot = data.oMf[fid]
            foot_tilt[side] = self._plane_rotvec(frame, foot.rotation)
            corners = np.array([foot.act(c) for c in kin.sole_corners[side]])
            wrench = self.wrenches.get(side)
            fz = abs(float(wrench.force[2])) if wrench is not None else 0.0
            foot_force[side] = fz
            # Support = the loaded feet, where the ZMP can be right now (an unloaded foot on the
            # ground would let the desired ZMP jump to it and over-excite the CoM admittance).
            # Without FT data the foot height decides.
            if wrench is not None:
                loaded = fz >= MIN_CONTACT_FORCE
            else:
                loaded = float(corners[:, 2].min() - ground_height) < contact_tol
            if not loaded:
                continue
            for c in corners:
                support_points.append(frame.to_plane(ref_inv.act(c))[:2])
            cop = cop_from_ankle_wrench(wrench.force, wrench.torque, ankle_to_sole) \
                if wrench is not None else None
            if cop is not None:
                cop_world = foot.act(np.array([cop[0], cop[1], -ankle_to_sole]))
                zmp_sum += frame.to_plane(ref_inv.act(cop_world))[:2] * fz
                fz_sum += fz
        hull = convex_hull(np.vstack(support_points)) if support_points else np.zeros((0, 2))
        zmp_valid = fz_sum >= MIN_CONTACT_FORCE
        zmp = zmp_sum / fz_sum if zmp_valid else np.zeros(2)

        tilt = self._plane_rotvec(frame, rot)
        tilt_rate = axial_to_plane(frame, self.ref_world.rotation.T @ self.base_omega_world)

        return BalanceEstimate(
            com=com[:2],
            com_vel=com_vel[:2],
            dcm=dcm,
            zmp=zmp,
            zmp_valid=bool(zmp_valid),
            support_hull=hull,
            tilt=tilt,
            tilt_rate=tilt_rate,
            foot_tilt=foot_tilt,
            foot_force=foot_force,
        )
