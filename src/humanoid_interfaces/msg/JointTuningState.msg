# Selected-joint tuning telemetry. Stamp is the joint-state sample time.
# Reference is taken from controller feedback, never inferred from the command.
std_msgs/Header header
builtin_interfaces/Time reference_stamp
string joint_name
float64 position_deg
float64 reference_position_deg
float64 velocity_deg_s
float64 velocity_upper_deg_s
float64 velocity_lower_deg_s
float64 effort_nm
float64 effort_upper_nm
float64 effort_lower_nm
