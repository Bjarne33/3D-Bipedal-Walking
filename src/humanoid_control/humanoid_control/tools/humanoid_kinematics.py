"""Pinocchio-based kinematics helper shared by the humanoid control nodes."""

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import pink
import pinocchio as pin
from pink.tasks import ComTask, FrameTask, PostureTask


@dataclass
class FootPose:
    """Foot pose expressed in the base (root joint) frame."""

    position: np.ndarray  # (3,) [x, y, z]
    rotation: np.ndarray  # (3, 3) rotation matrix
    quaternion: np.ndarray  # (4,) [x, y, z, w]


@dataclass
class KinematicsResult:
    """CoM position and foot poses in the base frame."""

    com: np.ndarray
    feet: Dict[str, FootPose]


def expand_xacro_to_urdf(xacro_path: str) -> str:
    """Expand a xacro file into a temporary URDF file.

    Args:
        xacro_path: Path to the xacro file.

    Returns:
        Path of the temporary URDF file; the caller removes it.
    """
    import xacro

    urdf_xml = xacro.process_file(xacro_path).toprettyxml(indent="  ")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as tmp:
        tmp.write(urdf_xml)
        return tmp.name


def sole_corners_from_urdf(urdf_path: str, link: str) -> Optional[np.ndarray]:
    """Find the bottom corners of a link's box collision geometry.

    Args:
        urdf_path: Path to the URDF file.
        link: Link name.

    Returns:
        Corners (4, 3) in the link frame, or None if the link has no box collision.
    """
    import xml.etree.ElementTree as ET

    for elem in ET.parse(urdf_path).getroot().iter("link"):
        if elem.get("name") != link:
            continue
        for collision in elem.iter("collision"):
            box = collision.find("geometry/box")
            if box is None:
                continue
            size = np.array([float(v) for v in box.get("size").split()])
            origin = collision.find("origin")
            xyz = np.zeros(3)
            rpy = np.zeros(3)
            if origin is not None:
                xyz = np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
                rpy = np.array([float(v) for v in origin.get("rpy", "0 0 0").split()])
            signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            corners = xyz + (0.5 * signs * size) @ pin.rpy.rpyToMatrix(rpy).T
            return corners[np.argsort(corners[:, 2])[:4]]
    return None


class HumanoidKinematics:
    """Loads the robot URDF via Pinocchio and provides kinematics.

    Provides forward kinematics (CoM and foot poses) and differential inverse
    kinematics (multi-task, solved with Pink) for walking control.
    """

    # Task costs: feet dominate, CoM is next, posture only regularizes.
    FOOT_POSITION_COST = 100.0
    FOOT_ORIENTATION_COST = 50.0
    COM_COST = 10.0
    # Keeps the torso upright so the CoM is shifted by the legs, not by leaning the torso.
    TORSO_ORIENTATION_COST = 10.0
    POSTURE_COST = 1e-3
    # Levenberg-Marquardt damping keeps the solution bounded near singularities.
    LM_DAMPING = 1e-2
    # Solvers tried in order; the first one available in qpsolvers is used.
    SOLVER_PREFERENCE = ("osqp", "quadprog", "clarabel", "cvxopt")
    # Largest joint velocity norm accepted from the solver [rad/s or m/s].
    MAX_VELOCITY = 20.0

    DEFAULT_FOOT_FRAMES = {"left": "left_foot_link", "right": "right_foot_link"}

    def __init__(self, urdf_path: str, foot_frames: Optional[Mapping[str, str]] = None):
        """Load the model and set up the inverse kinematics.

        Args:
            urdf_path: Path to the URDF file.
            foot_frames: Foot frame name per side; defaults to `DEFAULT_FOOT_FRAMES`.

        Raises:
            ValueError: If a foot frame is not found in the model.
        """
        self.urdf_path = urdf_path
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)

        self.joint_name_to_id = {name: jid for jid, name in enumerate(self.model.names)}

        self.foot_frames = dict(foot_frames or self.DEFAULT_FOOT_FRAMES)
        self.foot_frame_ids = {}
        for side, frame in self.foot_frames.items():
            if not self.model.existFrame(frame):
                raise ValueError(f"Foot frame '{frame}' ({side}) not found in {urdf_path}")
            self.foot_frame_ids[side] = self.model.getFrameId(frame)
        self.sole_corners = {
            side: sole_corners_from_urdf(urdf_path, frame)
            for side, frame in self.foot_frames.items()
        }

        self._init_ik()

    @classmethod
    def from_xacro(cls, xacro_path: str, foot_frames: Optional[Mapping[str, str]] = None):
        """Create the kinematics from a xacro file.

        Args:
            xacro_path: Path to the xacro file.
            foot_frames: Foot frame name per side; defaults to `DEFAULT_FOOT_FRAMES`.

        Returns:
            The kinematics instance.
        """
        urdf_path = expand_xacro_to_urdf(xacro_path)
        try:
            return cls(urdf_path, foot_frames)
        finally:
            os.remove(urdf_path)

    def set_joint_positions(self, names, positions) -> None:
        """Update the 1-DoF joints in the configuration; unknown names are ignored.

        Args:
            names: Joint names.
            positions: Joint positions, aligned with `names`.
        """
        for name, pos in zip(names, positions):
            jid = self.joint_name_to_id.get(name)
            if jid is None or self.model.joints[jid].nq != 1:
                continue
            self.q[self.model.idx_qs[jid]] = pos

    def forward_kinematics(self, joint_positions: Mapping[str, float]) -> KinematicsResult:
        """Compute CoM and foot poses (in the base frame) from joint positions.

        Args:
            joint_positions: Position per joint name.

        Returns:
            The CoM and foot poses.
        """
        self.set_joint_positions(joint_positions.keys(), joint_positions.values())
        return self.compute()

    def compute(self) -> KinematicsResult:
        """Run FK on the current configuration `self.q`.

        Returns:
            The CoM and foot poses in the base frame.
        """
        pin.forwardKinematics(self.model, self.data, self.q)
        pin.updateFramePlacements(self.model, self.data)
        com = np.array(pin.centerOfMass(self.model, self.data, self.q))

        feet = {}
        for side, fid in self.foot_frame_ids.items():
            placement = self.data.oMf[fid]
            feet[side] = FootPose(
                position=np.array(placement.translation),
                rotation=np.array(placement.rotation),
                quaternion=np.array(pin.Quaternion(placement.rotation).coeffs()),
            )
        return KinematicsResult(com=com, feet=feet)

    def joint_axes(self, names) -> np.ndarray:
        """Compute rotation axes of 1-DoF joints at the configuration `self.q`.

        Args:
            names: Joint names.

        Returns:
            Rotation axes, one row per joint, in the base frame.
        """
        data = self.model.createData()
        pin.computeJointJacobians(self.model, data, self.q)
        axes = []
        for name in names:
            jid = self.joint_name_to_id[name]
            jac = pin.getJointJacobian(self.model, data, jid, pin.ReferenceFrame.WORLD)
            axes.append(np.array(jac[3:, self.model.idx_vs[jid]]))
        return np.array(axes)

    def link_coms(self):
        """Return (joint_name, CoM position in base frame) per joint body.

        Call after `compute()` / `forward_kinematics()`.

        Returns:
            List of (joint_name, CoM position in the base frame) tuples.
        """
        return [
            (self.model.names[jid], self.data.oMi[jid].act(self.model.inertias[jid].lever))
            for jid in range(1, self.model.njoints)
        ]

    # ------------------------------------------------------------------
    # Inverse kinematics (Pink)
    # ------------------------------------------------------------------
    def _init_ik(self) -> None:
        """Create the Pink configuration and the IK tasks.

        The configuration at construction time is the nominal standing pose
        that the posture task regularizes towards.
        """
        import qpsolvers

        available = qpsolvers.available_solvers
        self.ik_solver = next((s for s in self.SOLVER_PREFERENCE if s in available), None)
        if self.ik_solver is None:
            raise RuntimeError("No QP solver available for Pink; run `pip install quadprog`")

        self.configuration = pink.Configuration(self.model, self.data, self.q.copy())

        # Both feet always have a task; which one is "stance" or "swing" is
        # decided per step by the frame names passed to compute_ik_step.
        self.tasks = {
            frame: FrameTask(
                frame,
                position_cost=self.FOOT_POSITION_COST,
                orientation_cost=self.FOOT_ORIENTATION_COST,
                lm_damping=self.LM_DAMPING,
            )
            for frame in self.foot_frames.values()
        }
        self.tasks["com"] = ComTask(cost=self.COM_COST, lm_damping=self.LM_DAMPING)
        self.tasks["torso"] = FrameTask(
            "torso_link",
            position_cost=0.0,
            orientation_cost=self.TORSO_ORIENTATION_COST,
            lm_damping=self.LM_DAMPING,
        )
        self.tasks["posture"] = PostureTask(cost=self.POSTURE_COST)

        self.tasks["posture"].set_target_from_configuration(self.configuration)
        self.tasks["torso"].set_target_from_configuration(self.configuration)
        # Hold both feet and the CoM where they are until targets are given.
        for frame in self.foot_frames.values():
            self.tasks[frame].set_target_from_configuration(self.configuration)
        self.tasks["com"].set_target_from_configuration(self.configuration)

    def reset_ik(self) -> None:
        """Restart IK from `self.q`, which also becomes the posture target."""
        self._init_ik()

    def compute_ik_step(self, stance_frame, stance_pose, swing_frame, swing_pose,
                        com_target, dt=0.02) -> np.ndarray:
        """Run one differential IK step and return the updated configuration.

        Args:
            stance_frame: Support foot frame name (e.g. 'left_foot_link').
            stance_pose: Desired `pin.SE3` of the stance foot (world frame).
            swing_frame: Swing foot frame name.
            swing_pose: Desired `pin.SE3` of the swing foot (world frame).
            com_target: Desired CoM position, (3,) array (world frame).
            dt: Integration time step [s].

        Returns:
            The configuration vector `self.configuration.q`. If the inputs are
            invalid or the QP fails, the configuration is left unchanged.
        """
        if stance_frame == swing_frame:
            raise ValueError("stance_frame and swing_frame must differ")
        for frame in (stance_frame, swing_frame):
            if frame not in self.tasks or frame in ("com", "torso", "posture"):
                raise ValueError(f"No foot task for frame '{frame}'")
        if not dt > 0.0:
            raise ValueError("dt must be positive")

        com_target = np.asarray(com_target, dtype=float).reshape(-1)
        poses_valid = all(
            np.all(np.isfinite(np.asarray(p.homogeneous)))
            for p in (stance_pose, swing_pose)
        )
        if com_target.shape != (3,) or not np.all(np.isfinite(com_target)) or not poses_valid:
            # Invalid target: keep the current configuration.
            return self.configuration.q

        # Pick up joint updates made through set_joint_positions / self.q.
        self.configuration.update(self.q)

        self.tasks[stance_frame].set_target(stance_pose)
        self.tasks[swing_frame].set_target(swing_pose)
        self.tasks["com"].set_target(com_target)

        try:
            velocity = pink.solve_ik(
                self.configuration, self.tasks.values(), dt=dt, solver=self.ik_solver
            )
        except Exception:  # QP infeasible or solver failure
            return self.configuration.q

        # Reject non-finite or excessive velocities (e.g. near a singularity).
        norm = np.linalg.norm(velocity)
        if not np.isfinite(norm):
            return self.configuration.q
        if norm > self.MAX_VELOCITY:
            velocity = velocity * (self.MAX_VELOCITY / norm)

        self.configuration.integrate_inplace(velocity, dt)
        self.q = self.configuration.q.copy()
        return self.configuration.q

    def inverse_kinematics(self, stance_frame, stance_pose, swing_frame, swing_pose,
                           com_target, dt=0.02, iterations=1) -> np.ndarray:
        """Solve IK towards the given targets using `iterations` Pink steps.

        Args:
            stance_frame: Support foot frame name.
            stance_pose: Desired `pin.SE3` of the stance foot.
            swing_frame: Swing foot frame name.
            swing_pose: Desired `pin.SE3` of the swing foot.
            com_target: Desired CoM position (3,).
            dt: Integration time step in seconds.
            iterations: Number of IK steps (at least 1).

        Returns:
            The resulting configuration vector.
        """
        q = self.configuration.q
        for _ in range(max(1, int(iterations))):
            q = self.compute_ik_step(
                stance_frame, stance_pose, swing_frame, swing_pose, com_target, dt
            )
        return q
