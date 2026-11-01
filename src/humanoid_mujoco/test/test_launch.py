"""Check the MuJoCo backend launch file and package metadata."""

import ast
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

from launch import LaunchContext
from launch.actions import DeclareLaunchArgument
from launch.utilities import perform_substitutions
from launch_ros.actions import Node

ROOT = Path(__file__).resolve().parents[1]
LAUNCH_FILE = ROOT / 'launch' / 'simulation.launch.py'


def _description():
    """Load the backend launch description.

    Returns:
        The launch description.
    """
    spec = importlib.util.spec_from_file_location('simulation', LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_launch_description()


def test_arguments_match_gazebo_backend():
    """Accept the Gazebo backend arguments plus the viewer switch."""
    description = _description()
    arguments = {
        entity.name: entity for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert set(arguments) == {'pause', 'world', 'use_sim_time', 'floating', 'gui'}
    context = LaunchContext()
    for name, value in {'pause': 'true', 'use_sim_time': 'true', 'floating': 'false',
                        'gui': 'true'}.items():
        assert perform_substitutions(context, arguments[name].default_value) == value
    nodes = sorted(
        (entity.node_package, entity.node_executable) for entity in description.entities
        if isinstance(entity, Node))
    assert nodes == [('humanoid_mujoco', 'mujoco_sim'),
                     ('robot_state_publisher', 'robot_state_publisher')]


def test_viewer_uses_x11_glfw_by_default(monkeypatch):
    """Select the X11 GLFW library unless the environment already chooses one."""
    simulator = next(entity for entity in _description().entities
                     if isinstance(entity, Node) and entity.node_executable == 'mujoco_sim')
    (name, value), = simulator.additional_env
    assert perform_substitutions(LaunchContext(), name) == 'PYGLFW_LIBRARY_VARIANT'
    monkeypatch.delenv('PYGLFW_LIBRARY_VARIANT', raising=False)
    assert perform_substitutions(LaunchContext(), value) == 'x11'
    monkeypatch.setenv('PYGLFW_LIBRARY_VARIANT', 'wayland')
    assert perform_substitutions(LaunchContext(), value) == 'wayland'


def test_dependencies_and_entry_point():
    """Declare every launched package and install the simulator executable."""
    dependencies = {
        entry.text for entry in ET.parse(ROOT / 'package.xml').findall('exec_depend')
    }
    packages = set()
    for call in ast.walk(ast.parse(LAUNCH_FILE.read_text())):
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
            if call.func.id == 'FindPackageShare':
                packages.add(ast.literal_eval(call.args[0]))
            elif call.func.id == 'Node':
                packages.update(ast.literal_eval(kw.value) for kw in call.keywords
                                if kw.arg == 'package')
    assert packages - {'humanoid_mujoco'} <= dependencies
    assert 'mujoco_sim = humanoid_mujoco.simulator:main' in (ROOT / 'setup.py').read_text()
