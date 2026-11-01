"""Verify the Gazebo backend model, launch sequence and Python style."""

import importlib.util
from pathlib import Path
import xml.etree.ElementTree as ET

from ament_flake8.main import main_with_errors
from ament_pep257.main import main
from launch import LaunchContext
from launch.actions import RegisterEventHandler
from launch.events.process import ProcessExited
from launch_ros.actions import Node
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


@pytest.mark.linter
def test_style():
    """Check formatting and Google-style docstrings for the backend package."""
    rc, errors = main_with_errors(argv=[str(ROOT)])
    assert rc == 0, '\n'.join(errors)
    assert main(argv=['--convention', 'google', '--', str(ROOT)]) == 0
