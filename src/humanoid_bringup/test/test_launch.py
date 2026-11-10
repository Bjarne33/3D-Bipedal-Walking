"""Check launch composition without starting a simulator or robot nodes."""

import ast
import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_index_python.packages import get_package_share_directory
from launch import LaunchContext, LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.utilities import perform_substitutions
from launch_ros.actions import Node


ROOT = Path(__file__).resolve().parents[1]
LAUNCH_FILE = ROOT / 'launch' / 'spawn_humanoid.launch.py'


def test_launch_description():
    """Check the existing arguments and visualization conditions around the backend include."""
    spec = importlib.util.spec_from_file_location('spawn_humanoid', LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    assert isinstance(description, LaunchDescription)
    arguments = {
        entity.name: entity for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert set(arguments) == {
        'simulator', 'floating', 'pause', 'world', 'use_sim_time', 'visualization', 'ui',
    }
    context = LaunchContext()
    defaults = {
        'simulator': 'gazebo', 'floating': 'false', 'pause': 'true', 'use_sim_time': 'true',
        'visualization': 'false', 'ui': 'true',
    }
    for name, value in defaults.items():
        assert perform_substitutions(context, arguments[name].default_value) == value
    nodes = {
        entity.node_executable: entity
        for entity in description.entities if isinstance(entity, Node)
    }
    for value in (None, 'false', 'true'):
        context = LaunchContext()
        if value is not None:
            context.launch_configurations['visualization'] = value
        arguments['visualization'].execute(context)
        for executable in ('rviz2', 'com_visualizer'):
            assert nodes[executable].condition.evaluate(context) == (value == 'true')
        for executable, node in nodes.items():
            if executable not in ('rviz2', 'com_visualizer', 'walk_manager_panel'):
                assert node.condition is None
    for value in (None, 'false', 'true'):
        context = LaunchContext()
        if value is not None:
            context.launch_configurations['ui'] = value
        arguments['ui'].execute(context)
        assert nodes['walk_manager_panel'].condition.evaluate(context) == (value != 'false')
    backends = [
        entity for entity in description.entities if isinstance(entity, IncludeLaunchDescription)
    ]
    assert len(backends) == 2
    for backend in backends:
        assert {name for name, value in backend.launch_arguments} == {
            'pause', 'world', 'use_sim_time', 'floating',
        }


def test_simulator_selection():
    """Include exactly the selected backend and default to its world file."""
    spec = importlib.util.spec_from_file_location('spawn_humanoid', LAUNCH_FILE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    arguments = {
        entity.name: entity for entity in description.entities
        if isinstance(entity, DeclareLaunchArgument)
    }
    assert arguments['simulator'].choices == ['gazebo', 'mujoco']
    backends = [
        entity for entity in description.entities if isinstance(entity, IncludeLaunchDescription)
    ]
    worlds = {'gazebo': 'world.sdf', 'mujoco': 'world.xml'}
    for simulator, world in worlds.items():
        share = Path(get_package_share_directory(f'humanoid_{simulator}'))
        context = LaunchContext()
        context.launch_configurations['simulator'] = simulator
        assert perform_substitutions(context, arguments['world'].default_value) == str(
            share / 'worlds' / world)
        included = [backend for backend in backends if backend.condition.evaluate(context)]
        assert len(included) == 1
        source = included[0].launch_description_source
        source.get_launch_description(context)
        assert source.location == str(share / 'launch' / 'simulation.launch.py')


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
        ('humanoid_control', 'humanoid_startup'),
        ('humanoid_control', 'humanoid_estimator'),
        ('humanoid_control', 'com_visualizer'),
        ('humanoid_control', 'humanoid_predefined_trajs'),
        ('humanoid_control', 'humanoid_walk'),
        ('humanoid_control', 'humanoid_rl_walk'),
        ('humanoid_control', 'humanoid_walk_manager'),
        ('humanoid_ui', 'walk_manager_panel'),
        ('rviz2', 'rviz2'),
    ])
