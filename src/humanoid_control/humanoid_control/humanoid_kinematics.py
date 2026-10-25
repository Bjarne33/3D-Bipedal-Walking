"""Pinocchio-based kinematics helper shared by the humanoid control nodes."""

import os
import tempfile
from dataclasses import dataclass
from typing import Dict, Mapping, Optional

import numpy as np
import pinocchio as pin


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

    Currently implements forward kinematics (CoM and foot poses). Inverse
    kinematics is expected to be added later and can reuse `self.model`,
    `self.data`, `self.q` and `self.foot_frame_ids`.
    """

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
