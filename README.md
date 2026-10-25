# 3D Bipedal Walking (ROS 2 Humble)

This workspace contains a simulated humanoid biped project using ROS 2 Humble and Gazebo.
It includes:

- Robot description and simulation launch files.
- Startup logic for controller readiness and initial pose.
- A walking test node that executes a single step sequence.
- CoM visualization markers in base and world frames.

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic (`gazebo_ros_pkgs`, `gazebo_ros2_control`)

## 1) Install ROS 2 Humble

Follow the official ROS 2 Humble installation guide for Ubuntu 22.04:

- https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html

After installation, source ROS 2:

```bash
source /opt/ros/humble/setup.bash
```

To avoid repeating this every terminal session, add it to your shell config:

```bash
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

## 2) Install Gazebo Classic

This project uses **Gazebo Classic** (`gazebo_ros`, `gazebo_ros2_control`)

Official docs:

- https://classic.gazebosim.org/
- https://github.com/ros-simulation/gazebo_ros_pkgs

Install Gazebo Classic with its ROS 2 packages and the project dependencies:

```bash
sudo apt update
sudo apt install -y \
	ros-humble-gazebo-ros-pkgs \
	ros-humble-gazebo-ros2-control \
	ros-humble-pinocchio \
	ros-humble-ros2-control \
	ros-humble-ros2-controllers \
	ros-humble-joint-state-publisher-gui \
	ros-humble-rviz2 \
	ros-humble-xacro
```

Inverse kinematics (`HumanoidKinematics.inverse_kinematics`) uses [Pink](https://github.com/stephane-caron/pink), which is installed via pip together with a QP solver:

```bash
pip install pin-pink quadprog
```

Optional but recommended for dependency installation:

```bash
sudo apt install -y python3-rosdep
sudo rosdep init || true
rosdep update
```

## 3) Clone the repository

Choose a parent folder and clone:

```bash
cd ~/Documents
git clone <YOUR-REPO-URL> 3D-Bipedal-Walking
```

If your workspace root is this repository, enter it:

```bash
cd 3D-Bipedal-Walking/3D-Bipedal-Walking
```

Install package dependencies from `src`:

```bash
rosdep install --from-paths src --ignore-src -r -y
```

## 4) Build the workspace

From workspace root (`3D-Bipedal-Walking`):

```bash
colcon build --symlink-install
```

## 5) Source the workspace

Source ROS 2 first, then this workspace overlay:

```bash
source /opt/ros/humble/setup.bash
source install/setup.bash
```

## 6) Run the project

Start the full simulation stack (Gazebo + robot spawn + controllers + RViz + control nodes):

```bash
ros2 launch humanoid_description spawn_humanoid.launch.py
# Optional: pin the base in the air so the robot cannot fall (useful for inspecting motions)
ros2 launch humanoid_description spawn_humanoid.launch.py floating:=true
```

Run URDF/RViz-only visualization (without Gazebo):

```bash
ros2 launch humanoid_description humanoid_rviz.launch.py
```

Trigger one walking step sequence from the walking node service:

```bash
ros2 service call /humanoid_walking/step_once std_srvs/srv/Trigger {}
ros2 service call /humanoid_walking/test_ik std_srvs/srv/Trigger {}   # IK check: left foot fixed, right foot and CoM moved by ik_test_* offsets
```

## Package Overview

### `humanoid_description`

Robot model and simulation assets.

- `urdf/`: Humanoid model as Xacro/URDF, including links, joints, inertia, transmissions, and `ros2_control` setup.
- `config/ros2_control_controllers.yaml`: Controller configuration.
- `worlds/world.sdf`: Gazebo world file.
- `rviz/urdf.rviz`: RViz configuration.
- `launch/spawn_humanoid.launch.py`: Full simulation bring-up:
	- Launches Gazebo.
	- Publishes `robot_description`.
	- Spawns the humanoid entity.
	- Spawns `joint_state_broadcaster` and `effort_position_controller`.
	- Starts RViz and control nodes from `humanoid_control`.
- `launch/humanoid_rviz.launch.py`: RViz/URDF preview workflow with joint state publisher.

### `humanoid_control`

Control and visualization nodes for startup and walking behavior.

- `humanoid_startup`:
	- Waits for controllers to become active.
	- Sends an initial standing pose to the trajectory controller.
	- Handles pause/unpause sequencing for startup.
- `humanoid_walking`:
	- Provides service `/humanoid_walking/step_once`.
	- Executes a single test step sequence with phases (shift, swing, place, settle).
	- Publishes `JointTrajectory` commands to `effort_position_controller`.
- `humanoid_walk` (model-based gait, `humanoid_control/gait/`):
	- Fixed-step footstep planner -> analytic LIPM/DCM CoM trajectory -> swing-foot trajectory -> Pink IK -> `JointTrajectory`.
	- Action `/humanoid_walk/walk` (`humanoid_interfaces/action/Walk`): `num_steps > 0` walks that many steps plus a closing step, `num_steps = 0` walks until canceled. Cancel or `/humanoid_walk/stop` closes the feet at the next step boundary.
	- Parameters in `humanoid_control/config/gait.yaml` (step length/time, swing height, DSP ratio, ...). Step width and CoM height are derived from the standing pose by default.
	- Before the first step the CoM is moved quasi-statically above the first stance foot (`shift_time`, quintic at rest) and held (`hold_time`); `shift_time: 0` disables this. Phases are reported as `shift`, `hold`, `init`, `step`, `final`.
	- Run: `ros2 launch humanoid_control walk.launch.py`, then e.g. `ros2 action send_goal /humanoid_walk/walk humanoid_interfaces/action/Walk "{num_steps: 4}"`.
	- Unit tests: `python3 -m pytest src/humanoid_control/test/test_gait.py`.
- `com_visualizer`:
	- Computes CoM using Pinocchio from robot model + joint states.
	- Publishes CoM markers and ground projection.
	- Optionally publishes world-frame CoM markers and `world -> base` TF.

## Typical Workflow

For every new terminal:

```bash
cd /path/to/3D-Bipedal-Walking
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Then launch:

```bash
ros2 launch humanoid_description spawn_humanoid.launch.py
```
