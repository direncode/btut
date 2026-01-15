# Configuration for BTUT simulation

# Basic parameters
int32 num_agents
float64 gamma          # Cooperation bonus
float64 tau            # Hub influence
float64 alpha          # Adaptation rate

# Network parameters
string network_type    # "barabasi_albert", "watts_strogatz", "erdos_renyi"
int32 network_param    # m for BA, k for WS, p for ER

# Convergence criteria
int32 max_iterations
float64 threshold

# Optional: initial strategy distribution
float64 initial_cooperation  # -1 for random, [0,1] for fixed
