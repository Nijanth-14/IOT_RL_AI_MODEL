import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

class IoTRoutingEnv(gym.Env):
    def __init__(self, num_nodes=20):
        super(IoTRoutingEnv, self).__init__()
        self.num_nodes = num_nodes
        
        # Action: AI selects which node to route the packet to next
        self.action_space = spaces.Discrete(self.num_nodes)
        
        # Observation: [Battery (0-100), Queue Size (0-inf), Link Quality (0-100), IsCurrentNode (0 or 1)]
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(self.num_nodes, 4), dtype=np.float32
        )
        self.graph = None 
        self.current_node = 0
        self.dest_node = self.num_nodes - 1

    def valid_action_mask(self):
        # Returns a binary mask for valid actions (1 if neighbor, 0 otherwise)
        mask = np.zeros(self.num_nodes, dtype=np.int8)
        if self.graph is not None:
            for neighbor in self.graph.neighbors(self.current_node):
                mask[neighbor] = 1
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. Create a larger, more complex topology
        self.graph = nx.connected_watts_strogatz_graph(self.num_nodes, k=6, p=0.2, seed=seed)
        
        # 2. Reset network state: 100% Battery, 0 Queue, Random Link Quality
        self.state = np.zeros((self.num_nodes, 4), dtype=np.float32)
        for i in range(self.num_nodes):
            self.state[i][0] = 100.0  
            self.state[i][1] = 0.0    
            self.state[i][2] = np.random.uniform(50.0, 100.0) 
            self.state[i][3] = 0.0 # Not the current node
            
        self.current_node = 0
        self.state[self.current_node][3] = 1.0 # Mark current node
        
        return self.state, {}

    def step(self, action):
        reward = 0.0
        terminated = False
        
        neighbors = list(self.graph.neighbors(self.current_node))
        
        if action not in neighbors:
            # Invalid move (with masking this shouldn't happen, but just in case)
            reward = -10.0 
            self.state[self.current_node][0] -= 5.0 
        else:
            # Valid move! Severe penalty if routing into congested nodes
            energy_cost = 1.0 + (self.state[action][1] * 3.0) # Look ahead at destination queue
            self.state[self.current_node][0] -= energy_cost
            
            # QoS penalty based on queue size (congestion)
            reward = - (1.0 + self.state[action][1] * 2.0)
            
            self.state[self.current_node][3] = 0.0
            self.current_node = action
            self.state[self.current_node][3] = 1.0
            
        # Win / Loss Conditions
        if self.current_node == self.dest_node:
            reward += 100.0 # Reached destination
            terminated = True
        elif self.state[self.current_node][0] <= 0:
            reward -= 100.0 # Drained battery
            terminated = True
            
        # Simulate heavy bursty network traffic
        for i in range(self.num_nodes):
            if np.random.rand() > 0.6: # More frequent traffic
                self.state[i][1] += np.random.uniform(1.0, 3.0) 
            # Slowly process queues
            self.state[i][1] = max(0.0, self.state[i][1] - 1.0)
            
        return self.state, reward, terminated, False, {}