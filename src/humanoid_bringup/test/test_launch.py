"""Check launch composition without starting Gazebo or robot nodes."""

import ast
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, RegisterEventHandler
from launch.utilities import perform_substitutions


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_FILE = ROOT / 'launch' / 'spawn_humanoid.launch.py'


def test_launch_description():
    """Check the existing launch arguments and controller event handlers are retained."""
    spec = importlib.util.spec_from_file_location('spawn_humanoid', LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    assert isinstance(description, LaunchDescription)
    arguments = {
        entity.name: entity for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert set(arguments) == {'floating', 'pause', 'world', 'use_sim_time'}
    context = LaunchContext()
    for name, value in {'floating': 'false', 'pause': 'true', 'use_sim_time': 'true'}.items():
        assert perform_substitutions(context, arguments[name].default_value) == value
    handlers = [
        entity for entity in description.entities if isinstance(entity, RegisterEventHandler)
    ]
    assert len(handlers) == 2


def test_launch_dependencies():
    """Require runtime dependencies for every launched node and package-share lookup."""
    dependencies = {
        entry.text for entry in ET.parse(ROOT / 'package.xml').findall('exec_depend')
    }
    tree = ast.parse(LAUNCH_FILE.read_text())
    packages = set()
    nodes = []
    for call in ast.walk(tree):
        if not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name):
            continue
        if call.func.id == 'FindPackageShare':
            packages.add(ast.literal_eval(call.args[0]))
        elif call.func.id == 'Node':
            keywords = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords
                        if kw.arg in ('package', 'executable')}
            packages.add(keywords['package'])
            nodes.append((keywords['package'], keywords['executable']))
    assert packages <= dependencies
    assert sorted(nodes) == sorted([
        ('robot_state_publisher', 'robot_state_publisher'),
        ('gazebo_ros', 'spawn_entity.py'),
        ('controller_manager', 'spawner'),
        ('controller_manager', 'spawner'),
        ('humanoid_control', 'humanoid_startup'),
        ('humanoid_control', 'humanoid_estimator'),
        ('humanoid_control', 'com_visualizer'),
        ('humanoid_control', 'humanoid_predefined_trajs'),
        ('rviz2', 'rviz2'),
    ])
