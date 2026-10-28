# Balance stabiliser state of humanoid_walk, published every control cycle on
# /humanoid_walk/balance_state. Positions are in the gait plane frame of the walk
# (x forward, y left, origin/orientation of the base at the start of the walk).
std_msgs/Header header
int32 step_index
string phase
bool enabled          # balance feedback active (false: open loop, estimate logged only)
bool estimate_valid   # measured base pose available
bool zmp_valid        # measured ZMP available (ankle FT, loaded feet)
bool saturated        # desired ZMP clamped to the support polygon
bool step_adjusted
float32 dcm_margin    # signed distance of the measured DCM to the support polygon edge (+ inside)
float32 ref_com_x
float32 ref_com_y
float32 meas_com_x
float32 meas_com_y
float32 ref_dcm_x
float32 ref_dcm_y
float32 meas_dcm_x
float32 meas_dcm_y
float32 ref_zmp_x
float32 ref_zmp_y
float32 meas_zmp_x
float32 meas_zmp_y
float32 des_zmp_x
float32 des_zmp_y
float32 com_offset_x
float32 com_offset_y
float32 tilt_x          # base rotation about the forward axis (roll) [rad]
float32 tilt_y          # base rotation about the left axis (pitch) [rad]
float32 ankle_rot_x     # stance-ankle foot-damping rotation [rad]
float32 ankle_rot_y
float32 torso_rot_x     # stance-hip torso rotation [rad]
float32 torso_rot_y
float32 step_adjust_x
float32 step_adjust_y
