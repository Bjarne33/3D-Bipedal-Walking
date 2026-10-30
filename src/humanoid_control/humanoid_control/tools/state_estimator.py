"""World-frame robot measurements and walking-frame balance projection."""

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


@dataclass
class RobotEstimate:
    """Robot measurements in world coordinates, independent of any walking reference."""

    base: pin.SE3
    base_omega: np.ndarray
    com: np.ndarray
    com_vel: np.ndarray
    feet: Dict[str, pin.SE3]
    foot_force: Dict[str, float]
    contact: Dict[str, bool]
    support_points: np.ndarray
    zmp: np.ndarray
    zmp_valid: bool


def axial_to_plane(frame, vec) -> np.ndarray:
    """Rotation vector / angular velocity -> right-handed [about forward, about left].

    The plane axes need not form a right-handed basis in base coordinates (this robot's
    left foot is at base +x while it walks along +y), so axial vectors get the handedness
    sign; a positive rotation about forward then always tilts the body to the right
    (towards -y) and a positive rotation about left tilts it forward.

    Args:
        frame: Plane frame of the gait.
        vec: Rotation vector or angular velocity (3,) in the base frame.

    Returns:
        The vector as [about forward, about left].
    """
    handed = np.sign(np.cross(frame.forward, frame.left) @ frame.up)
    return handed * frame.to_plane(np.asarray(vec, float))[:2]


def cop_from_ankle_wrench(force, torque, ankle_to_sole: float) -> Optional[np.ndarray]:
    """Centre of pressure (foot frame x, y) from the ankle wrench; None when unloaded.

    The ratio is independent of the wrench sign convention (joint wrench on the foot or
    ground reaction), so the joint feedback can be used directly. Foot inertia is ignored.

    Args:
        force: Ankle force (3,).
        torque: Ankle torque (3,).
        ankle_to_sole: Distance from the ankle to the sole in metres.

    Returns:
        CoP (x, y) in the foot frame, or None when the foot is unloaded.
    """
    fz = float(force[2])
    if abs(fz) < MIN_CONTACT_FORCE:
        return None
    d = float(ankle_to_sole)
    px = (-float(torque[1]) - float(force[0]) * d) / fz
    py = (float(torque[0]) - float(force[1]) * d) / fz
    return np.array([px, py])


class RobotStateEstimator:
    """Reconstruct world-frame robot state from base, joint and wrench measurements."""

    def __init__(self) -> None:
        """Create private measurement and forward-kinematics storage."""
        self.base_world: Optional[pin.SE3] = None
        self.base_vel_world = np.zeros(3)
        self.base_omega_world = np.zeros(3)
        self.base_time: Optional[float] = None
        self.joint_state: Dict[str, tuple] = {}
        self.wrenches: Dict[str, FootWrench] = {}
        self._wrench_sums: Dict[str, list] = {}
        self._wrench_lock = threading.Lock()
        self._model = None
        self._data = None
        self._q = None
        self._v = None

    # ------------------------------------------------------------------ inputs
    def set_base(self, base_world: pin.SE3, stamp: float, lin_vel=None, ang_vel=None) -> None:
        """Set the base pose in the world, optionally with its world-frame velocities.

        Args:
            base_world: Base pose in the world frame.
            stamp: Timestamp of the pose in seconds.
            lin_vel: Linear velocity (3,) in the world frame, or None.
            ang_vel: Angular velocity (3,) in the world frame, or None.
        """
        self.base_world = base_world
        self.base_time = float(stamp)
        if lin_vel is not None:
            self.base_vel_world = np.asarray(lin_vel, float)
        if ang_vel is not None:
            self.base_omega_world = np.asarray(ang_vel, float)

    def set_joint_state(self, names, positions, velocities=None, efforts=None) -> None:
        """Store the latest joint measurements.

        Args:
            names: Joint names.
            positions: Joint positions, aligned with `names`.
            velocities: Joint velocities, aligned with `names`; missing entries are 0.
            efforts: Joint efforts, aligned with `names`; missing entries are 0.
        """
        velocities = velocities if velocities is not None else []
        efforts = efforts if efforts is not None else []
        for i, name in enumerate(names):
            pos = float(positions[i]) if i < len(positions) else 0.0
            vel = float(velocities[i]) if i < len(velocities) else 0.0
            eff = float(efforts[i]) if i < len(efforts) else 0.0
            self.joint_state[name] = (pos, vel, eff)

    def set_wrench(self, side: str, force, torque) -> None:
        """Add a wrench sample; estimate() uses the mean of the samples since its last call.

        The sensors publish much faster than the control loop, and the stiff contact makes
        single samples noisy, so averaging over the loop period avoids aliasing.

        Args:
            side: 'left' or 'right'.
            force: Ankle force (3,).
            torque: Ankle torque (3,).
        """
        with self._wrench_lock:
            acc = self._wrench_sums.setdefault(side, [np.zeros(3), np.zeros(3), 0])
            acc[0] = acc[0] + np.asarray(force, float)
            acc[1] = acc[1] + np.asarray(torque, float)
            acc[2] += 1

    def _consume_wrenches(self) -> None:
        """Average the wrench samples collected since the last estimate and reset them."""
        with self._wrench_lock:
            for side, (force, torque, n) in self._wrench_sums.items():
                if n > 0:
                    self.wrenches[side] = FootWrench(force / n, torque / n)
            self._wrench_sums.clear()

    # ------------------------------------------------------------------ helpers
    def _bind(self, kin) -> None:
        """(Re)create the private Pinocchio data when the kinematics model changes.

        Args:
            kin: Kinematics object providing the model.
        """
        if self._model is not kin.model:
            self._model = kin.model
            self._data = kin.model.createData()
            self._q = pin.neutral(kin.model)
            self._v = np.zeros(kin.model.nv)

    # ------------------------------------------------------------------ estimate
    def estimate(self, kin, ground_height: float, contact_tol: float,
                 ankle_to_sole: float) -> Optional[RobotEstimate]:
        """Return the measured robot state without gait-dependent transformations.

        Args:
            kin: Kinematics object holding the current model.
            ground_height: Ground height in the world frame.
            contact_tol: Height tolerance below which a foot counts as in contact.
            ankle_to_sole: Distance from the ankle to the sole in metres.

        Returns:
            The robot estimate, or None before the base pose is known.
        """
        if self.base_world is None:
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

        support_points = []
        feet = {}
        foot_force = {}
        contact = {}
        zmp_sum = np.zeros(3)
        fz_sum = 0.0
        for side, fid in kin.foot_frame_ids.items():
            foot = data.oMf[fid]
            feet[side] = foot.copy()
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
            contact[side] = bool(loaded)
            if not loaded:
                continue
            for c in corners:
                support_points.append(c)
            cop = cop_from_ankle_wrench(wrench.force, wrench.torque, ankle_to_sole) \
                if wrench is not None else None
            if cop is not None:
                cop_world = foot.act(np.array([cop[0], cop[1], -ankle_to_sole]))
                zmp_sum += cop_world * fz
                fz_sum += fz
        zmp_valid = fz_sum >= MIN_CONTACT_FORCE
        zmp = zmp_sum / fz_sum if zmp_valid else np.zeros(3)

        return RobotEstimate(
            base=self.base_world.copy(),
            base_omega=self.base_omega_world.copy(),
            com=com_world,
            com_vel=com_vel_world,
            feet=feet,
            foot_force=foot_force,
            contact=contact,
            support_points=np.array(support_points).reshape(-1, 3),
            zmp=zmp,
            zmp_valid=bool(zmp_valid),
        )


class WalkStateProjection:
    """Project robot measurements into the walk-start frame and derive filtered DCM."""

    def __init__(self, reference: pin.SE3, alpha: float = 1.0) -> None:
        """Capture the walk reference and initialise the per-walk velocity filter."""
        self.reference = reference.copy()
        self.alpha = float(alpha)
        self._com_vel: Optional[np.ndarray] = None

    def project(self, state: RobotEstimate, frame, omega: float) -> BalanceEstimate:
        """Apply the original gait-plane projection, velocity filter and tilt conventions."""
        if self._com_vel is None or self.alpha >= 1.0:
            self._com_vel = state.com_vel.copy()
        else:
            self._com_vel = (1.0 - self.alpha) * self._com_vel + self.alpha * state.com_vel
        ref_inv = self.reference.inverse()
        com = frame.to_plane(ref_inv.act(state.com))[:2]
        com_vel = frame.to_plane(self.reference.rotation.T @ self._com_vel)[:2]
        points = np.array([
            frame.to_plane(ref_inv.act(p))[:2] for p in state.support_points
        ]).reshape(-1, 2)

        def tilt(rotation):
            return axial_to_plane(frame, pin.log3(self.reference.rotation.T @ rotation))

        return BalanceEstimate(
            com=com,
            com_vel=com_vel,
            dcm=com + com_vel / max(omega, 1e-6),
            zmp=frame.to_plane(ref_inv.act(state.zmp))[:2]
            if state.zmp_valid else np.zeros(2),
            zmp_valid=state.zmp_valid,
            support_hull=convex_hull(points) if len(points) else np.zeros((0, 2)),
            tilt=tilt(state.base.rotation),
            tilt_rate=axial_to_plane(frame, self.reference.rotation.T @ state.base_omega),
            foot_tilt={s: tilt(foot.rotation) for s, foot in state.feet.items()},
            foot_force=state.foot_force.copy(),
        )
