# 3D Bipedal Walking (ROS 2 Humble)

This workspace contains a simulated humanoid biped project using ROS 2 Humble, simulated in Gazebo
Classic (default) or MuJoCo.
It includes:

- Robot description and simulation launch files.
- Startup logic for controller readiness and initial pose.
- A joint-space test node (`humanoid_predefined_trajs`) that executes predefined step sequences.
- A model-based walking node (`humanoid_walk`) with closed-loop balance.
- An RL walking node (`humanoid_rl_walk`) that deploys the trained policy
  (`humanoid_control/policies/policy.npz`) in the MuJoCo backend.
- CoM visualization markers in base and world frames.
- Floating joint-controller PD tuning with an rqt panel and PlotJuggler 4.
- A learning-based walking controller (`rl/`, ROS-free) trained with curriculum reinforcement
  learning (Brax PPO on MuJoCo MJX); see [rl/README.md](rl/README.md).

Detailed system documentation is available as a [LaTeX project](latex/README.md), including a
VS Code LaTeX Workshop workspace and build recipe.

## Requirements

- Ubuntu 22.04
- ROS 2 Humble
- Gazebo Classic (`gazebo_ros_pkgs`, `gazebo_ros2_control`)
- MuJoCo 3.1.6 Python bindings (only for `simulator:=mujoco`)
- RL controller only: a separate Python virtual environment for `rl/` (MuJoCo >= 3.4, MJX,
  Playground, Brax, jax) and a CUDA GPU for training, e.g. Google Colab

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

The optional MuJoCo backend (`humanoid_mujoco`) uses the MuJoCo Python bindings, which have
no rosdep key. Install them via pip (tested with 3.1.6, which supports Python 3.10):

```bash
pip install mujoco==3.1.6
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

Start the simulation stack (simulator + robot spawn + controllers + control nodes).
Gazebo is the default simulator. RViz and the CoM visualizer are disabled by default:

```bash
ros2 launch humanoid_bringup spawn_humanoid.launch.py
# Optional: use MuJoCo instead of Gazebo
ros2 launch humanoid_bringup spawn_humanoid.launch.py simulator:=mujoco
# Optional: launch RViz and the CoM visualizer
ros2 launch humanoid_bringup spawn_humanoid.launch.py visualization:=true
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

Start the model-based gait node (with the simulation already running) and send a walk goal.
Wait until the robot has settled in the standing pose after startup (see Known issues):

```bash
ros2 launch humanoid_control walk.launch.py
ros2 action send_goal /humanoid_walk/walk humanoid_interfaces/action/Walk "{num_steps: 4}"
```

To deploy the trained RL policy instead, start the MuJoCo simulation, wait until the robot is at
rest, then start the RL walking node and send a goal (`forward_speed` in m/s, `duration` in s,
`duration: 0.0` walks until canceled):

```bash
ros2 launch humanoid_bringup spawn_humanoid.launch.py simulator:=mujoco
ros2 launch humanoid_control rl_walk.launch.py
ros2 action send_goal /humanoid_rl_walk/rl_walk humanoid_interfaces/action/RlWalk "{forward_speed: 0.15, duration: 10.0}"
```

## Joint PD tuning

`humanoid_tuning` runs an isolated, pinned-base experiment with either simulator. Do not run
normal bringup, walking nodes, or another trajectory publisher in the same ROS domain.
It tunes the existing backend controller, not a separate approximation of it.

Install the ROS UI dependencies through rosdep (or
`sudo apt install ros-humble-rqt-gui ros-humble-rqt-gui-py`), plus
[PlotJuggler 4](https://github.com/PlotJuggler/PlotJuggler) and its **ROS 2 Topic Subscriber**
and **ROS CDR Parser** marketplace extensions. PlotJuggler is an external desktop application,
not a pip/ROS dependency installed by this package. The installed `plotjuggler4` executable must
be on PATH. Source the built workspace before launching so its ROS message types are available.

```bash
colcon build --symlink-install --packages-up-to humanoid_tuning
source install/setup.bash
ros2 launch humanoid_tuning tuning.launch.py simulator:=gazebo
# Alternatively (use a separate session, not concurrently in the same domain):
ros2 launch humanoid_tuning tuning.launch.py simulator:=mujoco
```

The launch includes only the selected simulator backend and the standalone rqt panel; it
always uses `floating:=true`. The panel checks the published robot description and rejects an
unpinned robot or competing trajectory publishers. Opening the panel from the rqt menu defaults
to Gazebo; to attach it explicitly to an already-running, isolated floating backend, use:

```bash
ros2 run humanoid_tuning tuning_panel --simulator mujoco
```

### Experiment sequence

1. Select the test joint, edit its P/D values and click **Apply gains**. Confirmed applied
   values are displayed separately from unapplied edits. Gains use native **N m/rad** and
   **N m s/rad** units, require finite **P > 0, D >= 0**, and can only be applied between tests.
   Gazebo Humble 2.26 loads gains at configuration: the panel deactivates the controller,
   pauses physics while editing parameters, resumes physics, reconfigures, and reactivates.
   Configure/switch operations need the update loop running. MuJoCo commits accepted parameter
   batches to both actuator coefficients and trajectory-tracker gains under the simulation lock.
2. Edit the holding pose (all twelve joints start at zero) and click **Prepare**. It moves
   smoothly to that pose over **2 s**, then requires every joint to remain within **0.02 rad**
   position error and **0.02 rad/s** speed for **0.5 s**. Preparation times out after **15 s**.
   Timings and tolerances are editable; angles/speeds in the UI are **degrees/degrees per second**.
   Zero is the knee's upper limit: a positive knee target is not permitted.
3. Enter an **absolute** target angle and click **Start step**. A zero-time trajectory point
   jumps the controller's position reference, with zero desired velocity; it is not a short
   interpolated ramp. Other joints keep their holding references. Observation defaults to
   **5 s**, starting when backend feedback confirms the target and zero desired velocity,
   after which the target remains held. An unconfirmed step becomes a visible error after
   one wall second. Click **Prepare** before repeating.
4. **Stop (pause)** pauses physics and invalidates readiness. **Prepare** resumes physics and
   returns to the holding pose. Limit exceedances warn but do not pause physics. Missing/stale
   feedback or failed preparation shows an error and disables Start without automatically pausing.
   Service waits have wall-clock deadlines, including while simulation time is stopped.

The test joint is always plotted; check additional joints in the pose table to inspect coupling.
**PlotJuggler 4.0 requires manual subscriber setup:** its layout format cannot automatically
restore a live streaming source. The panel opens PlotJuggler, but does not pretend a saved
layout starts that connection:

1. In PlotJuggler, start **ROS 2 Topic Subscriber** and select the checked joints'
   `/humanoid_tuning/<joint>/state` topics.
2. Enable embedded/header timestamps, so the x-axis uses simulation time, not ROS receipt time.
3. Click **Copy layout path** in the panel, then use PlotJuggler's **Load layout** dialog and
   paste that path. The generated PJ4 layout creates a tab per selected joint, with separate
   position (actual/reference, deg), velocity (deg/s), and effort (N m) plots. Velocity and
   effort include the positive/negative URDF limits. Step rendering avoids smoothing references.
4. Changing plot selection or clicking **Open / reset plots** starts a fresh PlotJuggler window
   to prevent mixed histories; reconnect its subscriber and load the regenerated layout.
   The panel owns this window and closes it on shutdown. No response recordings are saved.

Telemetry uses `humanoid_interfaces/msg/JointTuningState`: scalar, unit-labelled fields, a
joint-state sample timestamp, and a separate `reference_stamp`. References come from actual
`/effort_position_controller/controller_state` feedback, never from a cached command. The worker
matches the most recent nonfuture reference within **10 ms** of each measured sample; it does not
fabricate missing reference values. Missing/alignment-stale streams invalidate readiness after
**1 wall second**. Acquisition follows the backend's 1 kHz joint stream independently of Qt's
20 Hz refresh; overloaded consumers may drop samples. No claim is made that displayed URDF
velocity limits are enforced by both physics engines.

### Export and limitations

**Export gains** creates a complete backend-native YAML in a **new user-selected file**, replacing
only confirmed applied P/D entries. It preserves all unrelated configuration values, including
untuned joints, Gazebo I/feed-forward gains, controller definitions and simulator settings.
Comments/formatting may change. Existing files and source/build/install/log destinations are
rejected. There is deliberately **no import/loading function or new loading launch argument**.
Export does not overwrite the backend's source configuration.

Floating tests still include gravity and inter-joint coupling. They do not validate stance
contacts, balance or walking. MuJoCo's armature and implicit damping differ from Gazebo's
explicit controller, so tune each backend independently; matching gain numbers need not produce
matching responses.

Tests (the real-backend tests create and stop their own isolated simulator):

```bash
python3 -m pytest src/humanoid_tuning/test src/humanoid_mujoco/test src/humanoid_gazebo/test
HUMANOID_TUNING_TEST_BACKEND=mujoco python3 -m pytest src/humanoid_tuning/test/test_tuning_integration.py
HUMANOID_TUNING_TEST_BACKEND=gazebo python3 -m pytest src/humanoid_tuning/test/test_tuning_integration.py
```

## Repository Structure

```text
src/
  humanoid_bringup/       System launch, composing the simulator backend and control nodes
  humanoid_gazebo/        Gazebo Classic model extensions, physics, controllers and launch
  humanoid_mujoco/        MuJoCo scene builder, ROS bridge node emulating the Gazebo interface, launch
  humanoid_description/   Simulator-independent Xacro model, RViz config and preview
  humanoid_interfaces/    Actions, services, RobotState, BalanceState and JointTuningState messages
  humanoid_tuning/        Floating PD experiments, rqt panel, native YAML export and PJ4 layouts
  humanoid_control/
    config/gait.yaml      Parameters of humanoid_walk
    config/estimator.yaml Sensor inputs, geometry and rate of humanoid_estimator
    config/rl_walk.yaml   Parameters of humanoid_rl_walk
    launch/walk.launch.py Starts humanoid_walk with gait.yaml
    launch/rl_walk.launch.py Starts humanoid_rl_walk with rl_walk.yaml
    policies/policy.npz   Trained RL policy exported from rl/ (default of humanoid_rl_walk)
    humanoid_control/
      nodes/              humanoid_startup, humanoid_predefined_trajs, humanoid_walk,
                          humanoid_rl_walk, humanoid_estimator, com_visualizer
      tools/              ROS-independent modules used by the nodes:
                            joints.py (joint order, standing pose), humanoid_kinematics.py (Pinocchio/Pink IK),
                            gait modules (footstep planner, LIPM/DCM, swing trajectory, walking pattern,
                            plan, balance, state estimator, support polygon, gait_params),
                            rl_policy.py (NumPy RL policy and observation)
    test/                 Unit tests and style checks
rl/                       Learning-based walking controller (not a colcon package, COLCON_IGNORE):
  humanoid_rl/            MJX environment, curriculum rewards, PPO curriculum driver, export
  scripts/export_mjcf.py  Generates the training MJCF from the description Xacro
  notebooks/              Google Colab training notebook
  tests/                  Unit and end-to-end pipeline tests
```

## Learning-based walking controller (`rl/`)

An independent walking policy trained with PPO (Brax) on MuJoCo MJX, using a three-phase reward
curriculum: posture and balance, weight shifting and foot clearance, low-speed forward walking.
Training targets a single GPU such as the free Google Colab tier
(`rl/notebooks/train_humanoid_colab.ipynb`). The package has its own virtual environment and
does not use ROS at runtime; the trained policy is exported as a NumPy `.npz`. The deployed
policy is `src/humanoid_control/policies/policy.npz`, run in ROS by `humanoid_rl_walk`. Setup, design (observations, actions, rewards, thresholds) and the tuning
checklist are in [rl/README.md](rl/README.md).

## Package Overview

### `humanoid_bringup`

System-level launch composition. Includes the selected simulator backend and starts the
existing control and visualization nodes.

- `launch/spawn_humanoid.launch.py`: Full simulation bring-up:
	- `simulator:=gazebo|mujoco` (default `gazebo`) selects `humanoid_gazebo` or `humanoid_mujoco`.
	  `pause`, `world`, `use_sim_time` and `floating` are forwarded to the backend; the `world`
	  default is the selected backend's world file.
	- Launches the simulator.
	- Publishes `robot_description`.
	- Spawns the humanoid entity.
	- Spawns `joint_state_broadcaster` and `effort_position_controller`.
	- Starts control nodes from `humanoid_control`.
	- Starts RViz and `com_visualizer` only when `visualization:=true` (default: `false`).
	- Starts one `humanoid_estimator`, shared by walking and visualization.

The simulation launch has moved from `humanoid_description` to `humanoid_bringup`;
its startup sequence is preserved, with visualization now opt-in. The model-based walking node
is still started separately using `ros2 launch humanoid_control walk.launch.py`.

### `humanoid_gazebo`

Gazebo Classic integration; depends on the shared description, not on control or bringup.

- `launch/simulation.launch.py`: Starts Gazebo, publishes the backend model, spawns the robot,
  then loads the joint-state broadcaster and trajectory controller in the existing order.
  Adds the parent of the installed `humanoid_description` share directory to
  `GAZEBO_MODEL_PATH` for both server and client, preserving existing paths so Gazebo can
  resolve the shared model's `package://humanoid_description/...` mesh visuals.
- `urdf/humanoid.xacro`: Includes the shared robot description and adds the unchanged Gazebo
  contact settings, ground-truth and ankle force/torque plugins.
- `urdf/ros2_control.xacro` and `urdf/transmissions/`: Gazebo hardware integration.
- `config/ros2_control_controllers.yaml`: Controller configuration.
- `config/gazebo_params.yaml`: Gazebo ROS node parameters (`/clock` published at 1000 Hz so sim-time control loops are not limited to the 10 Hz default).
- `worlds/world.sdf`: Gazebo world file.

The shared robot model remains authoritative for geometry, inertias and joint limits.

### `humanoid_mujoco`

MuJoCo integration; depends on the shared description, not on control or bringup. The node
`mujoco_sim` emulates the ROS interface of the Gazebo backend, so the control nodes, including
`humanoid_startup`, run unchanged.

- `humanoid_mujoco/model.py`: Imports the shared Xacro/URDF with MuJoCo's URDF importer (joint
  limits, damping, friction and effort limits are taken from it) and adds a free base joint (or
  welds `base` to the world with `floating:=true`), collision filtering, ankle force/torque
  sensors and one position actuator per joint. The robot is merged into the MJCF world.
  URDF visuals (including meshes with `package://` paths) are retained with their origins and
  scales, but never participate in collisions. The original collision shapes and inertias
  remain responsible for physics. MuJoCo supports binary STL, OBJ and MSH meshes; binary STL
  files must have at most 200,000 triangles (re-export more detailed CAD meshes at a lower
  resolution or use OBJ). Rebuild `humanoid_description` after adding mesh assets.
- `humanoid_mujoco/trajectory.py`: Joint-trajectory tracker emulating the
  `joint_trajectory_controller` (hold on activation, start from the current desired state, linear
  or cubic interpolation, hold the last point). Accelerations are ignored.
- `humanoid_mujoco/simulator.py` (`mujoco_sim`): Steps the physics in real time and provides:
	- `/clock` (every step), `/joint_states` (1000 Hz), `/odom` (100 Hz, `world -> base` ground
	  truth with the twist in `world`), `/left_foot_ft` and `/right_foot_ft` (500 Hz).
	- `/effort_position_controller/joint_trajectory` input.
	- `/effort_position_controller/controller_state` at the joint-state rate: reference,
	  feedback and error sampled/stamped before the physics step, matching the reference used
	  for that step. Joint states are published after stepping; applied efforts correspond to
	  that integration step. The tuning telemetry retains the separate reference timestamp.
	- Atomic runtime `gains.<joint>.p/d` parameter changes (P > 0, D >= 0); rejected batches
	  change neither parameters nor effective actuator/tracker gains.
	- `/pause_physics`, `/unpause_physics` and `/controller_manager/list_controllers`, which reports
	  `joint_state_broadcaster` and `effort_position_controller` as active.
	- An optional passive viewer. Its start camera looks at the robot's centre of mass from
	  `viewer_distance` (default 3 m), with the azimuth and elevation of `<visual><global>` in
	  the world file; "Align" in the viewer switches to MuJoCo's whole-scene view.
	  Visuals are shown by default; separate robot collision shapes are in geom group 3,
	  hidden by default and toggleable in the viewer. Links without visuals retain visible
	  collision shapes. The ground remains visible in group 0.
- `launch/simulation.launch.py`: Starts `robot_state_publisher` and `mujoco_sim`. Arguments:
  `pause`, `world`, `use_sim_time`, `floating` and `gui` (viewer, default `true`; not forwarded
  by bringup). It sets `PYGLFW_LIBRARY_VARIANT=x11` for `mujoco_sim` unless the variable is
  already set (see Known issues).
- `config/mujoco_sim.yaml`: Publish rates, emulated controller names, viewer start distance and
  PD gains. The gains are the RL training gains (`KP`/`KD` in `rl/humanoid_rl/constants.py`,
  kp 150-350, kd 4-10), much softer than the Gazebo gains, so that RL policies see their
  training dynamics. `humanoid_mujoco/model.py` likewise adds the training armature
  (0.02 kg m^2) to every actuated joint. With these gains the standing robot rests with about
  4-5 deg forward tilt from the static ankle droop; `humanoid_walk` and the predefined motions were
  tuned with the Gazebo gains and have not been revalidated in MuJoCo with them.
- `worlds/world.xml`: MJCF world (1 ms time step, `implicitfast` integrator, ground plane,
  viewer start direction).

The joint PD law `p (q_d - q) + d (v_d - q_dot)`, limited by the URDF effort limit, is the same
as in Gazebo (with different gains). It is realized with MuJoCo position actuators (`kp`, `kv`) because explicit
effort PD with these gains is unstable on MuJoCo's soft contacts; the world must therefore use
an implicit integrator (`implicit` or `implicitfast`). Run its tests with
`python3 -m pytest src/humanoid_mujoco/test`.

### `humanoid_description`

Simulator-independent robot model; does not depend on Gazebo, control or bringup.

- `urdf/`: Shared links, joints, inertias and geometry, without simulator plugins or hardware.
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
	- Reconstructs world-frame CoM/velocity, foot poses, loaded-foot support points and measured ZMP using a private Pinocchio state. Base pose/twist are simulator ground truth (Gazebo p3d or MuJoCo), not sensor-fused odometry; both backends express pose and twist in `world`.
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
- `humanoid_rl_walk` (RL policy deployment, `tools/rl_policy.py`; MuJoCo backend):
	- Loads the exported policy (`policy_path`, default: the installed `policies/policy.npz`) with NumPy only and checks its joint order against `tools/joints.py`.
	- Runs one policy step per `/humanoid_estimator/state` message (50 Hz = the policy's `ctrl_dt`). The observation mirrors the training environment: base-frame gyro and projected gravity (from the world-frame base pose and angular velocity), forward speed command, joint positions relative to the standing pose, joint velocities, last action and the gait clock (starting at phase 0 for every goal).
	- Publishes `standing pose + action_scale * action`, clipped to the URDF limits, as a single-point trajectory with `time_from_start = target_time_from_start` (default 0: applied at once, like training).
	- Action `/humanoid_rl_walk/rl_walk` (`humanoid_interfaces/action/RlWalk`): `forward_speed` (clamped to the trained range [0.05, 0.2] m/s with a warning) and `duration` (0 = until canceled). Feedback reports elapsed time, commanded and measured (CoM) forward speed and the gait phase. Cancel, the end of `duration` or `/humanoid_rl_walk/stop` command the standing pose at once.
	- Parameters in `config/rl_walk.yaml`; run with `ros2 launch humanoid_control rl_walk.launch.py` (optional `policy_path:=...`).
	- Headless MuJoCo trials (0.1-0.2 m/s, 8-10 s, plus cancel and stop): the robot walked forward without falling during the walk at about 0.6-0.8 of the commanded speed, with lateral and heading drift (the policy has no heading feedback; about 5-15 deg yaw per run). The abrupt switch to the standing pose causes 11-13 deg tilt transients; one of eight runs ended with the robot fallen (unmonitored, most likely at the switch).
- Foot force/torque sensing:
	- `libgazebo_ros_ft_sensor.so` is attached to both ankle-pitch joints and publishes `/left_foot_ft` and `/right_foot_ft`. In MuJoCo, force/torque sensors at the foot bodies publish the same topics.
	- These topics are consumed by `humanoid_estimator`, including when walking is idle or balance feedback is disabled.
- `com_visualizer`:
	- Consumes `/humanoid_estimator/state`; no longer owns a robot model or raw sensor subscriptions.
	- Publishes the same base-frame CoM markers and ground projection on `/com_marker`, and optional world-frame markers on `/com_world_marker`.
	- Does not broadcast TF. The old `model_states_topic`, `model_name`, and `publish_world_to_base_tf` visualizer parameters are removed; TF configuration belongs to the estimator.

## Known issues

- **MuJoCo viewer ignores mouse and keyboard on Wayland:** on a Wayland session (for example
  Ubuntu 22.04 with GNOME), pyGLFW (the pip `glfw` package, 2.7.0) loads its bundled
  GLFW 3.3.9 Wayland library. With it the viewer renders but receives no mouse or keyboard
  input: no camera rotation, zoom or button clicks (see pyGLFW issue
  [#81](https://github.com/FlorianRhiem/pyGLFW/issues/81), fixed by GLFW 3.4). The MuJoCo
  launch therefore sets `PYGLFW_LIBRARY_VARIANT=x11` for `mujoco_sim`, so the bundled X11
  library runs through XWayland, where input works. This also works on X11 sessions. A
  value already set in the environment takes precedence, for example to test native Wayland:
  `PYGLFW_LIBRARY_VARIANT=wayland ros2 launch ...`. To use a GLFW 3.4 build instead, set
  `PYGLFW_LIBRARY=/path/to/libglfw.so`, which pyGLFW prefers over the bundled libraries.

- **`AttributeError: _ARRAY_API not found` tracebacks (NumPy 2):** `humanoid_estimator` and
  `humanoid_walk` print these tracebacks at startup, together with a message that a module was
  compiled with NumPy 1.x. They come from importing Pink, which imports `qpsolvers`: the
  pip-installed `osqp`, `ecos` and `scs` wheels were built against NumPy 1.x and cannot be
  loaded with NumPy 2.x (observed with NumPy 2.2.6). `qpsolvers` skips these solvers and both
  nodes keep running. The IK uses the first available solver from
  `HumanoidKinematics.SOLVER_PREFERENCE` (`osqp`, `quadprog`, `clarabel`, `cvxopt`), so it falls
  back to `quadprog`. The tracebacks can be ignored as long as `quadprog` is installed. To remove
  them, install solver builds compatible with NumPy 2 or use NumPy < 2.
- **Robot not settled after startup:** `humanoid_startup` logs `Initial pose reached` once the
  joints reach the standing pose, but the robot as a whole still needs some time to settle in
  that pose. A walk goal sent right after startup starts from this unsettled state, and the
  robot can fall within the first steps. This is a limitation of the current startup procedure,
  not of the gait or balance controller. Until the startup sequence waits for the robot to
  settle, wait until the robot is at rest before sending a walk goal. This applies to both
  simulator backends.

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
