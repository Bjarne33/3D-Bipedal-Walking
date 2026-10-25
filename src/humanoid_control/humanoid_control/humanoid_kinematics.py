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
    com: np.ndarray
    feet: Dict[str, FootPose]


def expand_xacro_to_urdf(xacro_path: str) -> str:
    """Expand a xacro file into a temporary URDF file and return its path."""
    import xacro

    urdf_xml = xacro.process_file(xacro_path).toprettyxml(indent="  ")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".urdf", delete=False) as tmp:
        tmp.write(urdf_xml)
        return tmp.name


class HumanoidKinematics:
    """Loads the robot URDF via Pinocchio and provides kinematics.

    Provides forward kinematics (CoM and foot poses) and differential inverse
    kinematics (multi-task, solved with Pink) for walking control.
    """

    # Task costs: feet dominate, CoM is next, posture only regularizes.
    FOOT_POSITION_COST = 100.0
    FOOT_ORIENTATION_COST = 50.0
    COM_COST = 10.0
    POSTURE_COST = 1e-3
    # Levenberg-Marquardt damping keeps the solution bounded near singularities.
    LM_DAMPING = 1e-2
    # Solvers tried in order; the first one available in qpsolvers is used.
    SOLVER_PREFERENCE = ("osqp", "quadprog", "clarabel", "cvxopt")
    # Largest joint velocity norm accepted from the solver [rad/s or m/s].
    MAX_VELOCITY = 20.0

    DEFAULT_FOOT_FRAMES = {"left": "left_foot_link", "right": "right_foot_link"}

    def __init__(self, urdf_path: str, foot_frames: Optional[Mapping[str, str]] = None):
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

        self._init_ik()

    @classmethod
    def from_xacro(cls, xacro_path: str, foot_frames: Optional[Mapping[str, str]] = None):
        urdf_path = expand_xacro_to_urdf(xacro_path)
        try:
            return cls(urdf_path, foot_frames)
        finally:
            os.remove(urdf_path)

    def set_joint_positions(self, names, positions) -> None:
        """Update the 1-DoF joints in the configuration; unknown names are ignored."""
        for name, pos in zip(names, positions):
            jid = self.joint_name_to_id.get(name)
            if jid is None or self.model.joints[jid].nq != 1:
                continue
            self.q[self.model.idx_qs[jid]] = pos

    def forward_kinematics(self, joint_positions: Mapping[str, float]) -> KinematicsResult:
        """Compute CoM and foot poses (in the base frame) from joint positions."""
        self.set_joint_positions(joint_positions.keys(), joint_positions.values())
        return self.compute()

    def compute(self) -> KinematicsResult:
        """Run FK on the current configuration `self.q`."""
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

    def link_coms(self):
        """Return (joint_name, CoM position in base frame) per joint body.

        Call after `compute()` / `forward_kinematics()`.
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
        self.tasks["posture"] = PostureTask(cost=self.POSTURE_COST)

        self.tasks["posture"].set_target_from_configuration(self.configuration)
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
            if frame not in self.tasks or frame in ("com", "posture"):
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
        """Solve IK towards the given targets using `iterations` Pink steps."""
        q = self.configuration.q
        for _ in range(max(1, int(iterations))):
            q = self.compute_ik_step(
                stance_frame, stance_pose, swing_frame, swing_pose, com_target, dt
            )
        return q
