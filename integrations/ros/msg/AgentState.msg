# Message for agent state information

Header header

# Agent identification
int32 agent_id
string strategy          # "A" or "B"

# Position (for spatial coordination)
geometry_msgs/Pose pose

# Coordination metrics
float64 cooperation_rate
bool converged
int32 iteration

# Network properties
int32 degree
float64 centrality
bool is_hub
