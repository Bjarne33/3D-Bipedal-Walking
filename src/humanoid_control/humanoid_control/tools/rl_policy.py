"""NumPy deployment of the RL walking policy exported by `rl/humanoid_rl/export.py`.

The `.npz` holds the observation normalizer, the actor MLP (swish between layers) and metadata.
The deterministic action is `tanh(loc)` of the first `action_size` outputs, and the joint
target is `default_pose + action_scale * action`, clipped to the joint limits.

The observation (45 values) mirrors `humanoid_rl.env.HumanoidWalkEnv._get_obs` without noise.
All vectors are in the torso/IMU frame, which equals the `base` frame of the description:
    gyro (3), projected gravity (3), forward speed command (1), joint positions relative to the
    default pose (12), joint velocities (12), last action (12), gait clock sin/cos (2).
"""

from typing import Dict, Sequence, Tuple
import xml.etree.ElementTree as ET

import numpy as np

OBS_SIZE = 45


def quaternion_to_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    """Convert a quaternion to a rotation matrix.

    Args:
        quaternion_xyzw: Quaternion `(x, y, z, w)`; it is normalized first.

    Returns:
        3x3 rotation matrix.

    Raises:
        ValueError: If the quaternion has zero norm.
    """
    q = np.asarray(quaternion_xyzw, dtype=float)
    norm = np.linalg.norm(q)
    if norm == 0.0:
        raise ValueError('Quaternion has zero norm')
    x, y, z, w = q / norm
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def advance_phase(phase: float, dt: float, period: float) -> float:
    """Advance the gait clock phase by one control step.

    Args:
        phase: Current phase in [0, 1).
        dt: Control step [s].
        period: Gait cycle period [s].

    Returns:
        The new phase in [0, 1).
    """
    return float(np.mod(phase + dt / period, 1.0))


def clock_obs(phase: float) -> np.ndarray:
    """Return the clock observation `[sin(2 pi phase), cos(2 pi phase)]`.

    Args:
        phase: Gait clock phase in [0, 1).

    Returns:
        Array of shape (2,).
    """
    angle = 2.0 * np.pi * phase
    return np.array([np.sin(angle), np.cos(angle)])


def build_observation(base_rotation: np.ndarray, angular_velocity_world: Sequence[float],
                      command: float, joint_positions: Sequence[float],
                      joint_velocities: Sequence[float], default_pose: Sequence[float],
                      last_action: Sequence[float], phase: float) -> np.ndarray:
    """Assemble the policy observation from world-frame base and joint measurements.

    Args:
        base_rotation: World-from-base rotation matrix (3x3).
        angular_velocity_world: Base angular velocity in the world frame [rad/s].
        command: Forward speed command [m/s].
        joint_positions: Joint positions in policy joint order [rad].
        joint_velocities: Joint velocities in policy joint order [rad/s].
        default_pose: Default (standing) joint positions in policy joint order [rad].
        last_action: Previous policy action in [-1, 1].
        phase: Gait clock phase in [0, 1).

    Returns:
        Observation of shape (45,) as float32.
    """
    rotation = np.asarray(base_rotation, dtype=float)
    gyro = rotation.T @ np.asarray(angular_velocity_world, dtype=float)
    gravity = rotation.T @ np.array([0.0, 0.0, -1.0])
    parts = [
        gyro,
        gravity,
        [float(command)],
        np.asarray(joint_positions, dtype=float) - np.asarray(default_pose, dtype=float),
        np.asarray(joint_velocities, dtype=float),
        np.asarray(last_action, dtype=float),
        clock_obs(phase),
    ]
    return np.concatenate([np.ravel(p) for p in parts]).astype(np.float32)


def clamp_speed(speed: float, low: float, high: float) -> Tuple[float, bool]:
    """Clamp a forward speed command to the trained range.

    Args:
        speed: Requested speed [m/s].
        low: Lowest trained speed [m/s].
        high: Highest trained speed [m/s].

    Returns:
        Tuple of the clamped speed and whether it was changed.
    """
    clamped = float(min(max(speed, low), high))
    return clamped, clamped != float(speed)


def joint_limits_from_urdf(urdf: str, joint_names: Sequence[str]) -> Tuple[np.ndarray,
                                                                           np.ndarray]:
    """Read the position limits of the given joints from a URDF.

    Args:
        urdf: URDF XML.
        joint_names: Joints whose limits are returned, in this order.

    Returns:
        Tuple of lower and upper limit arrays [rad].

    Raises:
        ValueError: If a joint or its limit is missing.
    """
    limits: Dict[str, Tuple[float, float]] = {}
    for joint in ET.fromstring(urdf).findall('joint'):
        limit = joint.find('limit')
        if limit is not None and 'lower' in limit.attrib and 'upper' in limit.attrib:
            limits[joint.get('name')] = (float(limit.get('lower')), float(limit.get('upper')))
    missing = [name for name in joint_names if name not in limits]
    if missing:
        raise ValueError(f'URDF has no position limits for {missing}')
    lower, upper = zip(*(limits[name] for name in joint_names))
    return np.array(lower), np.array(upper)


class RlPolicy:
    """Deterministic actor MLP with observation normalization, loaded from an `.npz`."""

    def __init__(self, path: str) -> None:
        """Load and validate the exported policy.

        Args:
            path: Path of the `.npz` written by `humanoid_rl.export.export_npz`.

        Raises:
            ValueError: If metadata required for deployment is missing or inconsistent.
        """
        data = np.load(path)
        required = ('obs_mean', 'obs_std', 'num_layers', 'action_size', 'default_pose',
                    'joint_names', 'action_scale', 'ctrl_dt', 'gait_period')
        missing = [key for key in required if key not in data.files]
        if missing:
            raise ValueError(f'Policy {path!r} lacks {missing} (export it with an env config)')
        self.obs_mean = data['obs_mean'].astype(np.float32)
        self.obs_std = data['obs_std'].astype(np.float32)
        num_layers = int(data['num_layers'])
        self.layers = [(data[f'w{i}'].astype(np.float32), data[f'b{i}'].astype(np.float32))
                       for i in range(num_layers)]
        self.action_size = int(data['action_size'])
        self.joint_names = [str(name) for name in data['joint_names']]
        self.default_pose = data['default_pose'].astype(float)
        self.action_scale = float(data['action_scale'])
        self.ctrl_dt = float(data['ctrl_dt'])
        self.gait_period = float(data['gait_period'])
        self.phase = int(data['phase']) if 'phase' in data.files else None
        if self.obs_mean.shape != (OBS_SIZE,) or self.obs_std.shape != (OBS_SIZE,):
            raise ValueError(f'Expected {OBS_SIZE} observations, got {self.obs_mean.shape}')
        if self.layers[0][0].shape[0] != OBS_SIZE:
            raise ValueError('First layer does not match the observation size')
        if self.layers[-1][1].shape[-1] != 2 * self.action_size:
            raise ValueError('Last layer does not produce 2 * action_size outputs')
        if len(self.joint_names) != self.action_size or \
                self.default_pose.shape != (self.action_size,):
            raise ValueError('joint_names and default_pose must match action_size')
        if self.ctrl_dt <= 0.0 or self.gait_period <= 0.0:
            raise ValueError('ctrl_dt and gait_period must be positive')

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        """Compute the deterministic action for an observation.

        Args:
            obs: Observation of shape (45,) (or a batch of them).

        Returns:
            Action in [-1, 1] of shape (action_size,) (or a batch of them).
        """
        x = (np.asarray(obs, np.float32) - self.obs_mean) / self.obs_std
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i < len(self.layers) - 1:
                # Swish; exp overflows to inf for very negative inputs, giving the correct 0.
                with np.errstate(over='ignore'):
                    x = x / (1.0 + np.exp(-x))
        return np.tanh(x[..., :self.action_size])

    def joint_targets(self, action: np.ndarray, lower: np.ndarray,
                      upper: np.ndarray) -> np.ndarray:
        """Convert an action into joint position targets.

        Args:
            action: Policy action in [-1, 1].
            lower: Lower joint limits in policy joint order [rad].
            upper: Upper joint limits in policy joint order [rad].

        Returns:
            Joint position targets [rad].
        """
        return np.clip(self.default_pose + self.action_scale * np.asarray(action), lower, upper)
