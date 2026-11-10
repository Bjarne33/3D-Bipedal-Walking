"""Verify the Gazebo backend model, launch sequence and Python style."""

import importlib.util
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_flake8.main import main_with_errors
from ament_pep257.main import main
from launch import LaunchContext
from launch.actions import AppendEnvironmentVariable, ExecuteProcess, IncludeLaunchDescription
from launch.actions import RegisterEventHandler
from launch.events.process import ProcessExited
from launch.utilities import perform_substitutions
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
import pytest
import xacro


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('floating', ['false', 'true'])
def test_model(floating):
    """Check the backend adds plugins without modifying the shared robot geometry.

    Args:
        floating: Whether to pin the robot base.
    """
    shared_path = ROOT.parent / 'humanoid_description' / 'urdf' / 'humanoid.xacro'
    shared = ET.fromstring(xacro.process_file(
        str(shared_path), mappings={'floating': floating},
    ).toxml())
    gazebo = ET.fromstring(xacro.process_file(
        str(ROOT / 'urdf' / 'humanoid.xacro'), mappings={'floating': floating},
    ).toxml())
    for tag in ('link', 'joint'):
        assert [ET.tostring(e) for e in shared.findall(tag)] == [
            ET.tostring(e) for e in gazebo.findall(tag)
        ]
    assert not shared.findall('gazebo')
    assert not shared.findall('ros2_control')
    assert not shared.findall('transmission')
    assert len(gazebo.findall('transmission')) == 12
    assert gazebo.find('ros2_control/hardware/plugin').text == (
        'gazebo_ros2_control/GazeboSystem'
    )
    plugins = gazebo.findall('gazebo/plugin')
    assert {plugin.attrib['name'] for plugin in plugins} == {
        'gazebo_ros2_control', 'base_ground_truth', 'left_foot_ft', 'right_foot_ft',
    }
    parameters = gazebo.find('gazebo/plugin/parameters').text
    assert Path(parameters).read_bytes() == (
        ROOT / 'config/ros2_control_controllers.yaml'
    ).read_bytes()


def test_controller_sequence():
    """Check robot spawn still triggers the broadcasters and trajectory controller in order."""
    spec = importlib.util.spec_from_file_location(
        'gazebo_simulation', ROOT / 'launch' / 'simulation.launch.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    nodes = [e for e in description.entities if isinstance(e, Node)]
    assert [node.node_executable for node in nodes] == [
        'robot_state_publisher', 'spawn_entity.py',
    ]
    handlers = [e.event_handler for e in description.entities
                if isinstance(e, RegisterEventHandler)]
    assert len(handlers) == 2
    context = LaunchContext()
    spawn_exit = ProcessExited(
        action=nodes[1], name='spawn_entity', cmd=[], cwd=None, env=None, pid=1, returncode=0,
    )
    assert handlers[0].matches(spawn_exit)
    broadcaster = handlers[0].handle(spawn_exit, context)[0]
    broadcaster_exit = ProcessExited(
        action=broadcaster, name='spawner', cmd=[], cwd=None, env=None, pid=2, returncode=0,
    )
    assert handlers[1].matches(broadcaster_exit)
    controller = handlers[1].handle(broadcaster_exit, context)[0]
    assert broadcaster.node_executable == controller.node_executable == 'spawner'


@pytest.mark.parametrize('existing_path', [None, '', '/custom/models:/other/models'])
def test_mesh_search_path(monkeypatch, existing_path):
    """Check both Gazebo processes can find installed meshes while retaining user paths."""
    if existing_path is None:
        monkeypatch.delenv('GAZEBO_MODEL_PATH', raising=False)
    else:
        monkeypatch.setenv('GAZEBO_MODEL_PATH', existing_path)
    spec = importlib.util.spec_from_file_location(
        'gazebo_simulation', ROOT / 'launch' / 'simulation.launch.py',
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    description = module.generate_launch_description()
    context = LaunchContext()
    for action in description.entities:
        if isinstance(action, IncludeLaunchDescription):
            break
        if isinstance(action, AppendEnvironmentVariable):
            action.execute(context)

    share = Path(FindPackageShare('humanoid_description').perform(context))
    paths = os.environ.get('GAZEBO_MODEL_PATH', '').split(os.pathsep)
    assert share.parent in [Path(path).resolve() for path in paths if path]
    if existing_path:
        assert paths[:2] == existing_path.split(os.pathsep)

    model = ET.fromstring(xacro.process_file(
        str(ROOT / 'urdf' / 'humanoid.xacro'),
    ).toxml())
    meshes = model.findall('.//visual/geometry/mesh')
    assert meshes
    gazebo_share = Path(FindPackageShare('gazebo_ros').perform(context))
    for filename in ('gzserver.launch.py', 'gzclient.launch.py'):
        spec = importlib.util.spec_from_file_location(
            'gazebo_process', gazebo_share / 'launch' / filename,
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        processes = [
            entity for entity in module.generate_launch_description().entities
            if isinstance(entity, ExecuteProcess)
        ]
        assert processes
        for process in processes:
            environment = {
                perform_substitutions(context, key): perform_substitutions(context, value)
                for key, value in process.additional_env
            }
            roots = environment['GAZEBO_MODEL_PATH'].split(os.pathsep)
            for mesh in meshes:
                uri = mesh.attrib['filename']
                assert uri.startswith('package://')
                relative_path = uri.removeprefix('package://')
                assert any(
                    (Path(root) / relative_path).is_file() for root in roots if root
                ), uri


@pytest.mark.linter
def test_style():
    """Check formatting and Google-style docstrings for the backend package."""
    rc, errors = main_with_errors(argv=[str(ROOT)])
    assert rc == 0, '\n'.join(errors)
    assert main(argv=['--convention', 'google', '--', str(ROOT)]) == 0
