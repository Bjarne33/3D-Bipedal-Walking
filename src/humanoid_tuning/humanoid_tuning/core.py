# Copyright 2026 Bjarne
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure validation, settling and one-way backend-native gain export."""

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Optional
import xml.etree.ElementTree as ET

import yaml


@dataclass(frozen=True)
class JointLimits:
    """URDF position, speed and effort limits in native SI units."""

    lower: float
    upper: float
    velocity: float
    effort: float


def parse_limits(urdf):
    """Extract bounded revolute joint limits.

    Args:
        urdf: Expanded robot XML.

    Returns:
        Joint name to validated limits.

    Raises:
        ValueError: If limits are absent, invalid or nonfinite.
    """
    result = {}
    for joint in ET.fromstring(urdf).findall('joint'):
        if joint.get('type') != 'revolute':
            continue
        limit = joint.find('limit')
        if limit is None:
            raise ValueError(f'Missing limits for {joint.get("name")}')
        values = [float(limit.attrib[key]) for key in ('lower', 'upper', 'velocity', 'effort')]
        if (not all(math.isfinite(v) for v in values) or values[0] > values[1]
                or values[2] <= 0 or values[3] <= 0):
            raise ValueError(f'Invalid limits for {joint.get("name")}')
        result[joint.attrib['name']] = JointLimits(*values)
    if not result:
        raise ValueError('Robot description has no bounded revolute joints')
    return result


def validate_position(value, limit):
    """Check a native-radian target against a joint's URDF bounds.

    Args:
        value: Target angle in radians.
        limit: Joint limits.

    Raises:
        ValueError: If the value is nonfinite or outside the inclusive bounds.
    """
    if not math.isfinite(value) or not limit.lower <= value <= limit.upper:
        raise ValueError(f'Position must be within [{limit.lower:g}, {limit.upper:g}] rad')


def validate_gains(p_gain, d_gain):
    """Validate the common backend PD gain range.

    Args:
        p_gain: Proportional gain in N m/rad.
        d_gain: Derivative gain in N m s/rad.

    Raises:
        ValueError: If either gain is invalid.
    """
    if (not math.isfinite(p_gain) or not math.isfinite(d_gain)
            or p_gain <= 0 or d_gain < 0):
        raise ValueError('Gains must be finite, with P > 0 and D >= 0')


def native_gains(config, simulator):
    """Find the native gain mapping without altering other configuration.

    Args:
        config: Complete backend YAML mapping.
        simulator: Backend name.

    Returns:
        Native joint gain mapping.

    Raises:
        ValueError: If the backend is unknown.
    """
    if simulator not in ('gazebo', 'mujoco'):
        raise ValueError(f'Unknown simulator: {simulator}')
    node = 'effort_position_controller' if simulator == 'gazebo' else 'mujoco_sim'
    return config[node]['ros__parameters']['gains']


def export_config(destination, baseline, simulator, applied):
    """Exclusively create a complete native YAML with confirmed applied gains.

    Args:
        destination: New user-selected path; existing paths are rejected.
        baseline: Original complete configuration, retained unchanged.
        simulator: Selected backend.
        applied: Confirmed joint to (P, D) mapping.

    Raises:
        ValueError: If a gain is invalid or a joint is absent.
        OSError: If the new file cannot be created or written.
    """
    config = deepcopy(baseline)
    gains = native_gains(config, simulator)
    for name, (p_gain, d_gain) in applied.items():
        validate_gains(p_gain, d_gain)
        if name not in gains:
            raise ValueError(f'Unknown joint in export: {name}')
        gains[name].update(p=float(p_gain), d=float(d_gain))
    payload = yaml.safe_dump(config, sort_keys=False)
    path = Path(destination).expanduser()
    if any(part in ('src', 'build', 'install', 'log') for part in path.resolve().parts):
        raise ValueError('Export outside source/build/install/log directories')
    with path.open('x', encoding='utf-8') as stream:
        stream.write(payload)


@dataclass
class Settling:
    """Continuous all-joint position/velocity settling check using simulation time."""

    window: float = 0.5
    position_tolerance: float = 0.02
    velocity_tolerance: float = 0.02
    since: Optional[float] = None
    previous: Optional[float] = None

    def update(self, stamp, positions, velocities, targets):
        """Check an entire sample, restarting the window on failure or clock reset.

        Args:
            stamp: Sample simulation time in seconds.
            positions: Measured radians in target order.
            velocities: Measured radians/second in target order.
            targets: Holding angles in radians.

        Returns:
            Whether the complete window has met both criteria.
        """
        reset = self.previous is not None and stamp < self.previous
        self.previous = stamp
        valid = (len(positions) == len(velocities) == len(targets)
                 and all(math.isfinite(q) and math.isfinite(v)
                         and abs(q - target) <= self.position_tolerance
                         and abs(v) <= self.velocity_tolerance
                         for q, v, target in zip(positions, velocities, targets)))
        if reset or not valid:
            self.since = None
        if valid and self.since is None:
            self.since = stamp
        return valid and stamp - self.since >= self.window
