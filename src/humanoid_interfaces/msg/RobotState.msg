# Geometric quantities are in header.frame_id (world), except com_base and joint_state.
# Without base odometry, only com_base and joint_state are valid.
std_msgs/Header header
string base_frame_id
bool base_valid
builtin_interfaces/Time base_stamp
sensor_msgs/JointState joint_state
builtin_interfaces/Time left_wrench_stamp
builtin_interfaces/Time right_wrench_stamp
bool left_wrench_available
bool right_wrench_available

geometry_msgs/Pose base_pose
geometry_msgs/Vector3 base_angular_velocity
geometry_msgs/Point com
geometry_msgs/Point com_base
geometry_msgs/Vector3 com_velocity
geometry_msgs/Pose left_foot_pose
geometry_msgs/Pose right_foot_pose
float64 left_normal_force
float64 right_normal_force
bool left_contact
bool right_contact
geometry_msgs/Point[] support_points
geometry_msgs/Point zmp
bool zmp_valid
