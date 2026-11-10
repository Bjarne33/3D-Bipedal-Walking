# State of the walk manager (humanoid_walk_manager), published latched on
# /humanoid_walk_manager/state whenever it changes and periodically.
string STARTUP=STARTUP          # waiting for humanoid_startup to report the standing pose
string SETTLING=SETTLING        # waiting until the robot is at rest
string IDLE=IDLE                # standing at rest, ready for a walk request
string WALKING=WALKING          # model-based walk (/humanoid_walk/walk) running
string RL_WALKING=RL_WALKING    # RL policy walk (/humanoid_rl_walk/rl_walk) running
string STOPPING=STOPPING        # stop requested, waiting for the walk to end
string FALLEN=FALLEN            # robot fell; walking is disabled until it stands again

std_msgs/Header header
string state
bool walk_enabled      # /humanoid_walk_manager/walk is accepted now
bool rl_walk_enabled   # /humanoid_walk_manager/rl_walk is accepted now
bool stop_enabled      # /humanoid_walk_manager/stop is accepted now
string message         # human-readable detail on the current state
