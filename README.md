# 3D Bipedal Walking

A ROS 2 Humble workspace for simulating and controlling a simplified 12-joint,
three-dimensional biped. Gazebo Classic is the default simulator; an optional MuJoCo
backend provides the same ROS interfaces so the control stack can run on either.

The workspace includes a shared robot model, simulation backends, state estimation,
predefined joint-space motions, a model-based walking controller with closed-loop
balance, an exported reinforcement-learning (RL) policy, and an rqt walk-manager
panel. The RL policy is intended for MuJoCo. This is a simulation and research
project: base odometry is simulator ground truth, and walking is not safety-certified.


## Installation and setup

### Requirements

- Ubuntu 22.04 and ROS 2 Humble.
- Gazebo Classic and `gazebo_ros` / `gazebo_ros2_control` for the default backend.
- Python 3 dependencies for inverse kinematics: Pink and a QP solver such as
  `quadprog`.
- Optional MuJoCo backend: Python package `mujoco==3.1.6`.
- Optional RL training: a separate Python environment; a CUDA-capable GPU is
  recommended for training. RL training dependencies are not used by ROS runtime.
- Optional PD tuning UI: rqt and PlotJuggler 4 with its ROS 2 Topic Subscriber and
  ROS CDR Parser extensions.

Install ROS 2 Humble for Ubuntu 22.04 using the
[official instructions](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html),
then source it in each terminal:

```bash
source /opt/ros/humble/setup.bash
```

Clone the repository and install ROS package dependencies:

```bash
git clone https://github.com/Bjarne33/Bjarne33-3D-Bipedal-Walking.git
cd Bjarne33-3D-Bipedal-Walking

sudo apt update
sudo apt install -y python3-rosdep python3-pip \
  ros-humble-gazebo-ros-pkgs \
  ros-humble-gazebo-ros2-control \
  ros-humble-pinocchio \
  ros-humble-ros2-control \
  ros-humble-ros2-controllers \
  ros-humble-joint-state-publisher-gui \
  ros-humble-rviz2 \
  ros-humble-xacro

# Run rosdep init only once per machine, then update its package index.
sudo rosdep init
rosdep update
rosdep install --from-paths src --ignore-src -r -y

# Install Python-only control dependencies into the Python environment used by ROS.
python3 -m pip install pin-pink quadprog
```

For MuJoCo, also install its Python bindings in the environment used to launch ROS:

```bash
python3 -m pip install mujoco==3.1.6
```

Build and source the workspace overlay:

```bash
colcon build --symlink-install
source install/setup.bash
```

Source both ROS and the workspace overlay in every new terminal. After changing
package metadata, launch files, or installed configuration/assets, rebuild the
workspace. The `rl/` training project is excluded from `colcon`; see its guide for
its independent virtual environment.

## Project structure

```text
src/
  humanoid_description/  Shared Xacro robot model and RViz-only preview
  humanoid_gazebo/       Gazebo Classic world, plugins, contacts and controllers
  humanoid_mujoco/       MuJoCo scene, simulator bridge and controller emulation
  humanoid_bringup/      Selects a backend and composes the full simulation
  humanoid_control/      Startup, estimation, walking, gait and balance code/config
  humanoid_interfaces/   Actions, services and messages shared by ROS packages
  humanoid_ui/           rqt walk-manager panel
  humanoid_tuning/       Isolated joint PD experiments and telemetry/export UI
rl/                      ROS-free MuJoCo MJX/Brax PPO training and policy export
docs/latex/               Detailed LaTeX system documentation
```

The shared model is authoritative for robot geometry, inertias and joint limits.
Simulator-specific physics and hardware configuration belongs in the backend
packages. `humanoid_bringup` starts one shared estimator and the control components;
the estimator publishes `RobotState`, which is consumed by the model-based walker,
RL walker and walk manager.

## Usage

### Start the simulation

From the sourced workspace:

```bash
# Gazebo Classic (default)
ros2 launch humanoid_bringup spawn_humanoid.launch.py

# Optional backend and launch settings
ros2 launch humanoid_bringup spawn_humanoid.launch.py simulator:=mujoco
ros2 launch humanoid_bringup spawn_humanoid.launch.py visualization:=true
ros2 launch humanoid_bringup spawn_humanoid.launch.py floating:=true
ros2 launch humanoid_bringup spawn_humanoid.launch.py ui:=false
```

Bringup starts the simulator, controllers, startup pose, estimator, predefined-motion
node, both walking nodes, walk manager, and (by default) its rqt panel. The simulation
starts paused; startup handles controller readiness and pause/unpause sequencing.
`floating:=true` pins the base in the air for inspecting joint motion; it is not a
walking test. The MuJoCo backend viewer is enabled by default.

Wait for startup to report the initial pose **and** for the robot to settle at rest
before requesting a walk. Reaching the joint-position tolerance alone does not mean
the whole robot has settled.

### Walk using the manager

The panel provides **Walk**, **RL Walk**, and **Stop**. In a separate sourced
terminal, the same controls are available as services:

```bash
ros2 service call /humanoid_walk_manager/walk std_srvs/srv/Trigger "{}"
ros2 service call /humanoid_walk_manager/rl_walk std_srvs/srv/Trigger "{}"
ros2 service call /humanoid_walk_manager/stop std_srvs/srv/Trigger "{}"
ros2 topic echo /humanoid_walk_manager/state
```

The manager waits for startup and settling, then enables walking when the robot is
idle. Its Walk button requests continuous model-based walking; RL Walk uses the
configured forward speed (default 0.15 m/s) on the shipped policy. Stop cancels the
active action. The model-based walker closes the feet at a step boundary; the RL
walker commands its standing pose immediately. Do not send competing motion commands.

### Send an action goal directly

For a finite model-based walk, send a goal only when the manager is not already
running a walk:

```bash
ros2 action send_goal /humanoid_walk/walk \
  humanoid_interfaces/action/Walk "{num_steps: 4}" --feedback
```

Positive `num_steps` walks that many steps followed by a closing step; `num_steps: 0`
walks continuously until canceled or stopped. The RL action is intended for MuJoCo;
its speed is clamped to 0.05–0.2 m/s, and `duration: 0.0` means run until canceled:

```bash
ros2 action send_goal /humanoid_rl_walk/rl_walk \
  humanoid_interfaces/action/RlWalk \
  "{forward_speed: 0.15, duration: 10.0}" --feedback
```

Normal bringup already launches both action servers. For custom compositions without
bringup, the individual launch files are
`humanoid_control walk.launch.py` and `humanoid_control rl_walk.launch.py`; start
exactly one estimator as well as a simulator backend.

### Other workflows

Predefined joint-space inspection motions:

```bash
ros2 service call /humanoid_predefined_trajs/step_once std_srvs/srv/Trigger "{}"
ros2 service call /humanoid_predefined_trajs/step_n \
  humanoid_interfaces/srv/StepN "{num_steps: 4}"
```

Preview the model in RViz without starting a simulator:

```bash
ros2 launch humanoid_description humanoid_rviz.launch.py
```

For isolated pinned-base PD experiments, see the tuning and PlotJuggler instructions
in the [system guide](docs/latex/README.md). Do not run tuning or another trajectory
publisher concurrently with normal bringup.

Useful state checks include:

```bash
ros2 topic hz /humanoid_estimator/state
ros2 topic echo --once /humanoid_walk_manager/state
ros2 topic echo --once /humanoid_walk/balance_state
```

## Validation and limitations

Run focused ROS package tests from the workspace root with ROS sourced:

```bash
python3 -m pytest src/humanoid_control/test/test_gait.py
python3 -m pytest src/humanoid_mujoco/test
```

For RL training and its tests, create and use the separate `rl/.venv` described in
[rl/README.md](rl/README.md). Walking behavior depends on simulator, CPU load,
startup settling, model and configuration; do not infer performance from a passing
unit test or assume Gazebo and MuJoCo results are interchangeable. The system guide
records known issues, validation scope and the detailed ROS interfaces.
