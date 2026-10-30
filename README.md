# 3D Bipedal Walking (ROS 2 Humble)

This workspace contains a simulated humanoid biped project using ROS 2 Humble and Gazebo.
It includes:

- Robot description and simulation launch files.
- Startup logic for controller readiness and initial pose.
- A joint-space test node (`humanoid_predefined_trajs`) that executes predefined step sequences.
- A model-based walking node (`humanoid_walk`) with closed-loop balance.
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
ros2 launch humanoid_bringup spawn_humanoid.launch.py
# Optional: pin the base in the air so the robot cannot fall (useful for inspecting motions)
ros2 launch humanoid_bringup spawn_humanoid.launch.py floating:=true
```

Run URDF/RViz-only visualization (without Gazebo):

```bash
ros2 launch humanoid_description humanoid_rviz.launch.py
```

Trigger the joint-space test motions:

```bash
ros2 service call /humanoid_predefined_trajs/step_once std_srvs/srv/Trigger {}
ros2 service call /humanoid_predefined_trajs/step_n humanoid_interfaces/srv/StepN "{num_steps: 4}"   # n steps, alternating right/left
```

Start the model-based gait node (with the simulation already running) and send a walk goal:

```bash
ros2 launch humanoid_control walk.launch.py
ros2 action send_goal /humanoid_walk/walk humanoid_interfaces/action/Walk "{num_steps: 4}"
```

## Repository Structure

```text
src/
  humanoid_bringup/       Full simulation launch, composing description and control packages
  humanoid_description/   Xacro model, Gazebo world, controller config, RViz config and preview
  humanoid_interfaces/    Walk action, StepN service, BalanceState and RobotState messages
  humanoid_control/
    config/gait.yaml      Parameters of humanoid_walk
    config/estimator.yaml Sensor inputs, geometry and rate of humanoid_estimator
    launch/walk.launch.py Starts humanoid_walk with gait.yaml
    humanoid_control/
      nodes/              humanoid_startup, humanoid_predefined_trajs, humanoid_walk,
                          humanoid_estimator, com_visualizer
      tools/              ROS-independent modules used by the nodes:
                            joints.py (joint order, standing pose), humanoid_kinematics.py (Pinocchio/Pink IK),
                            gait modules (footstep planner, LIPM/DCM, swing trajectory, walking pattern,
                            plan, balance, state estimator, support polygon, gait_params)
    test/                 Unit tests and style checks
```

## Package Overview

### `humanoid_bringup`

System-level launch composition. Depends on the description and control packages and the
Gazebo, controller and visualization components needed to launch the simulation.

- `launch/spawn_humanoid.launch.py`: Full simulation bring-up:
	- Launches Gazebo.
	- Publishes `robot_description`.
	- Spawns the humanoid entity.
	- Spawns `joint_state_broadcaster` and `effort_position_controller`.
	- Starts RViz and control nodes from `humanoid_control`.
	- Starts one `humanoid_estimator`, shared by walking and visualization.

The simulation launch has moved from `humanoid_description` to `humanoid_bringup`;
its arguments, defaults and startup sequence are unchanged. The model-based walking node
is still started separately using `ros2 launch humanoid_control walk.launch.py`.

### `humanoid_description`

Robot model and simulation assets; does not depend on the control or bringup packages.

- `urdf/`: Humanoid model as Xacro/URDF, including links, joints, inertia, transmissions, and `ros2_control` setup.
- `config/ros2_control_controllers.yaml`: Controller configuration.
- `config/gazebo_params.yaml`: Gazebo ROS node parameters (`/clock` published at 1000 Hz so sim-time control loops are not limited to the 10 Hz default).
- `worlds/world.sdf`: Gazebo world file.
- `rviz/urdf.rviz`: RViz configuration.
- `launch/humanoid_rviz.launch.py`: RViz/URDF preview workflow with joint state publisher.

### `humanoid_control`

Control and visualization nodes for startup and walking behavior.

- `humanoid_startup`:
	- Waits for controllers to become active.
	- Sends an initial standing pose to the trajectory controller.
	- Handles pause/unpause sequencing for startup.
- `humanoid_predefined_trajs` (joint-space test motions):
	- Provides service `/humanoid_predefined_trajs/step_once`.
	- Executes a single test step sequence with phases (shift, swing, place, settle).
	- Provides service `/humanoid_predefined_trajs/step_n` (`humanoid_interfaces/srv/StepN`): repeats the test step `num_steps` times, alternating right and left foot (left steps are the mirrored right step).
	- Publishes `JointTrajectory` commands to `effort_position_controller`.
- `humanoid_estimator`:
	- Owns `/odom`, `/joint_states`, `/left_foot_ft`, and `/right_foot_ft` subscriptions.
	- Reconstructs world-frame CoM/velocity, foot poses, loaded-foot support points and measured ZMP using a private Pinocchio state. Base pose/twist are Gazebo p3d ground truth, not sensor-fused odometry; the plugin expresses both pose and twist in `world`.
	- Publishes `/humanoid_estimator/state` (`humanoid_interfaces/msg/RobotState`) at 50 Hz. The message includes source timestamps, measured joint states, base-frame CoM, contact/force information and ZMP validity. Without base odometry, `base_valid=false`: only joint states and base-frame CoM are valid, preserving base-only visualization and open-loop walking.
	- Averages ankle wrenches over each estimator period (nominally the same 50 Hz as walking). Loaded-foot threshold remains 30 N; absent wrench data still uses the sole-height contact fallback.
	- Broadcasts dynamic `world -> base` TF using the odometry source timestamp. With `floating:=true`, bringup disables this broadcaster because `robot_state_publisher` supplies the fixed transform. The RViz-only preview retains its static publisher.
	- Parameters formerly owned by the walker (`odom_topic`, `left_ft_topic`, `right_ft_topic`, `ground_height`, `contact_tolerance`) now live in `config/estimator.yaml`. Keep its `ankle_to_sole` aligned with `gait.yaml`, and its `update_rate_hz` aligned with the walking rate when preserving the nominal averaging window.
	- Estimator and consumers warn about stale sources (`state_timeout_sec`, default 0.2 s); cached measurements remain in use, as before. This refactor does not add a new emergency-stop policy or synchronized sensor fusion.
	- Started by simulation bringup, **not** by `walk.launch.py`, to avoid duplicate estimators and TF publishers. For custom launch compositions, start exactly one estimator (`ros2 run humanoid_control humanoid_estimator`, with simulation time/configuration as appropriate).
- `humanoid_walk` (model-based gait, `humanoid_control/tools/`):
	- Fixed-step footstep planner -> analytic LIPM/DCM CoM trajectory -> swing-foot trajectory -> Pink IK (feet, CoM, upright torso orientation) -> `JointTrajectory`.
	- Closed-loop stabilizer between pattern generation and IK (`tools/state_estimator.py`, `tools/balance.py`), on by default (`balance_enabled: true`; `false` gives the open-loop gait, the estimate is still published):
		- Consumes `/humanoid_estimator/state` instead of raw sensors. Captures the walk-start base pose, projects measurements into the gait plane, filters CoM velocity and computes DCM `xi = c + c_dot / omega`. The velocity filter resets at each walk, and `omega` still follows the gait CoM height. Support points are projected before computing the 2D convex hull, preserving the original frame conventions.
		- DCM tracking: `p_des = p_ref + k_xi (xi - xi_ref)`, clamped into the support polygon minus `balance_zmp_margin`.
		- CoM admittance: the IK CoM target is shifted by an offset integrated from `-k_com (c - c_ref) - k_zmp (p_des - p_meas)` (leaky, limited). In double support the ZMP term is scaled by `balance_zmp_dsp_scale`, since the measured ZMP then also reflects internal leg forces.
		- Torso feedback: measured base tilt is fed back (`balance_k_torso_p/d`) through the stance hip in single support (both hips in double support).
		- Stance-ankle damping (`balance_k_ankle_roll/pitch`) is available but off by default (no benefit in Gazebo tuning). Step adjustment (`step_adjust_*`) is only computed and published, not applied to the footstep plan.
		- `zmp_lateral: 0.08` places the single-support reference ZMP 4 cm inside the stance foot centre, which reduces the lateral rocking.
	- Publishes `/humanoid_walk/balance_state` (`humanoid_interfaces/msg/BalanceState`: reference/measured CoM, DCM, ZMP, desired ZMP, saturation, offsets) for debug/monitoring.
	- Prior Gazebo validation, before the estimator-node refactor (headless, balance defaults): `num_steps: 6` and continuous walking cancelled after 10 s (9 steps, feet closed) each completed 3/3 runs without a fall; CoM error RMS 1.5-2.9 cm, base roll/pitch < 7 deg. The open-loop gait falls after about 3 s (step 1-2). The refactor has numerical and synthetic-sensor ROS regression coverage, but these physical simulation trials have not been repeated for it.
	- Action `/humanoid_walk/walk` (`humanoid_interfaces/action/Walk`): `num_steps > 0` walks that many steps plus a closing step, `num_steps = 0` walks until canceled. Cancel or `/humanoid_walk/stop` closes the feet at the next step boundary.
	- Parameters in `humanoid_control/config/gait.yaml` (step length/time, swing height, DSP ratio, balance gains/limits, ...). Step width and CoM height are derived from the standing pose by default.
	- Before the first step the CoM is moved quasi-statically above the first stance foot (`shift_time`, quintic at rest) and held (`hold_time`); `shift_time: 0` disables this. Phases are reported as `shift`, `hold`, `init`, `step`, `final`.
	- Run: `ros2 launch humanoid_control walk.launch.py`, then e.g. `ros2 action send_goal /humanoid_walk/walk humanoid_interfaces/action/Walk "{num_steps: 4}"`.
	- Unit tests: `python3 -m pytest src/humanoid_control/test`.
	- Parameters (`gait.yaml`) override the `GaitParams` defaults; `step_width` and `com_height` <= 0 are derived from the standing pose.
- Foot force/torque sensing:
	- `libgazebo_ros_ft_sensor.so` is attached to both ankle-pitch joints and publishes `/left_foot_ft` and `/right_foot_ft`.
	- These topics are consumed by `humanoid_estimator`, including when walking is idle or balance feedback is disabled.
- `com_visualizer`:
	- Consumes `/humanoid_estimator/state`; no longer owns a robot model or raw sensor subscriptions.
	- Publishes the same base-frame CoM markers and ground projection on `/com_marker`, and optional world-frame markers on `/com_world_marker`.
	- Does not broadcast TF. The old `model_states_topic`, `model_name`, and `publish_world_to_base_tf` visualizer parameters are removed; TF configuration belongs to the estimator.

## Known issues

- **Intermittent walk rate-timer destruction race (ROS 2 Humble):** `humanoid_walk`
  creates a rate for each walk and calls `destroy_rate()` at the end. With its
  `MultiThreadedExecutor`, a queued timer callback can try to use the timer after
  destruction was requested, raising
  `rclpy._rclpy_pybind11.InvalidHandle: cannot use Destroyable because destruction was requested`.
  This can interrupt executor processing around walk completion. It was observed during
  repeated synthetic-sensor action regression tests; the affected timer lifecycle predates
  the estimator refactor and is deliberately left unchanged. This is not a claim about
  Gazebo walking performance. The action lifecycle regression in
  `test/test_state_estimator.py` is marked non-strict `xfail` **only for `InvalidHandle`**:
  passing runs are reported as XPASS, and other exception types still fail normally.
  A future dedicated fix should address timer lifetime/queued callbacks and revalidate
  finite walking, cancellation, and the stop service.

## Typical Workflow

For every new terminal:

```bash
cd /path/to/3D-Bipedal-Walking
source /opt/ros/humble/setup.bash
source install/setup.bash
```

Then launch:

```bash
ros2 launch humanoid_bringup spawn_humanoid.launch.py
```
