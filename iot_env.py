import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import numpy as np

class IoTRoutingEnv(gym.Env):
    """
    Energy and QoS-Aware Intelligent Routing Environment for IoT Networks.
    
    The agent must route packets from source (node 0) to destination (last node)
    while navigating dynamic congestion, link failures, and energy constraints.
    
    Observation per node (5 features):
        - Battery level (0-100)
        - Queue size / congestion (0-100)
        - Link quality (0-100)  
        - Is current node (0 or 1)
        - Is destination node (0 or 1)
    
    The RL agent's advantage over shortest-path algorithms:
        - It learns to AVOID congested nodes (high queue)
        - It learns to AVOID low-battery nodes (preserving network lifetime)
        - It adapts to dynamic link failures in real-time
    """
    
    def __init__(self, num_nodes=20, traffic_intensity=0.5, failure_prob=0.05, 
                 max_steps=50):
        super(IoTRoutingEnv, self).__init__()
        self.num_nodes = num_nodes
        self.traffic_intensity = traffic_intensity  # Controls congestion severity
        self.failure_prob = failure_prob            # Probability of link failure per step
        self.max_steps = max_steps
        
        # Action: AI selects which node to route the packet to next
        self.action_space = spaces.Discrete(self.num_nodes)
        
        # Observation: 5 features per node
        self.observation_space = spaces.Box(
            low=0, high=100, shape=(self.num_nodes, 5), dtype=np.float32
        )
        self.graph = None
        self.original_edges = None  # Store original edges for link failure/recovery
        self.current_node = 0
        self.dest_node = self.num_nodes - 1
        self.steps = 0
        self.visited_nodes = set()

    def valid_action_mask(self):
        """Returns a binary mask for valid actions (1 if neighbor, 0 otherwise)."""
        mask = np.zeros(self.num_nodes, dtype=np.int8)
        if self.graph is not None:
            for neighbor in self.graph.neighbors(self.current_node):
                mask[neighbor] = 1
        return mask

    def _get_obs(self):
        """Build the observation from current state."""
        obs = np.zeros((self.num_nodes, 5), dtype=np.float32)
        for i in range(self.num_nodes):
            obs[i][0] = self.battery[i]
            obs[i][1] = min(self.queue[i], 100.0)  # Cap for observation space
            obs[i][2] = self.link_quality[i]
            obs[i][3] = 1.0 if i == self.current_node else 0.0
            obs[i][4] = 1.0 if i == self.dest_node else 0.0
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Create a connected small-world IoT network
        self.graph = nx.connected_watts_strogatz_graph(
            self.num_nodes, k=6, p=0.3, seed=seed
        )
        self.original_edges = list(self.graph.edges())
        
        # Initialize node states
        self.battery = np.full(self.num_nodes, 100.0, dtype=np.float32)
        self.queue = np.zeros(self.num_nodes, dtype=np.float32)
        self.link_quality = np.random.uniform(50.0, 100.0, size=self.num_nodes).astype(np.float32)
        
        # Pre-congest some nodes along the shortest path to make it interesting
        try:
            sp = nx.shortest_path(self.graph, source=0, target=self.dest_node)
            for node in sp[1:-1]:  # Congest intermediate nodes on shortest path
                self.queue[node] = np.random.uniform(3.0, 8.0)
                self.battery[node] = np.random.uniform(20.0, 60.0)
        except nx.NetworkXNoPath:
            pass
            
        self.current_node = 0
        self.steps = 0
        self.visited_nodes = {0}
        
        return self._get_obs(), {}

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        self.steps += 1
        
        neighbors = list(self.graph.neighbors(self.current_node))
        
        if action not in neighbors:
            # Invalid move — should not happen with action masking
            reward = -20.0
            self.battery[self.current_node] -= 5.0
        else:
            next_node = action
            
            # ---- Energy Cost (QoS-aware) ----
            # Cost depends on: queue congestion at next node + link quality
            congestion_factor = 1.0 + (self.queue[next_node] * 0.5)
            quality_factor = max(0.1, (100.0 - self.link_quality[next_node]) / 100.0)
            energy_cost = congestion_factor + quality_factor * 2.0
            self.battery[self.current_node] -= energy_cost
            
            # ---- Reward Shaping ----
            # Base step penalty
            reward = -1.0
            
            # Congestion penalty: heavily penalize routing into congested nodes
            reward -= self.queue[next_node] * 1.5
            
            # Energy awareness: penalize routing through low-battery nodes
            if self.battery[next_node] < 30.0:
                reward -= (30.0 - self.battery[next_node]) * 0.5
            
            # Loop penalty: discourage revisiting nodes
            if next_node in self.visited_nodes:
                reward -= 3.0
            
            # Progress reward: reward getting closer to destination
            try:
                old_dist = nx.shortest_path_length(self.graph, self.current_node, self.dest_node)
                new_dist = nx.shortest_path_length(self.graph, next_node, self.dest_node)
                if new_dist < old_dist:
                    reward += 5.0  # Getting closer
                elif new_dist > old_dist:
                    reward -= 2.0  # Moving away
            except nx.NetworkXNoPath:
                reward -= 10.0
            
            # Move the packet
            self.current_node = next_node
            self.visited_nodes.add(next_node)
        
        # ---- Win / Loss Conditions ----
        if self.current_node == self.dest_node:
            # Bonus scaled by efficiency: fewer steps + more remaining energy = higher reward
            efficiency_bonus = max(0, 50 - self.steps) * 2.0
            energy_bonus = np.mean(self.battery) * 0.5
            reward += 100.0 + efficiency_bonus + energy_bonus
            terminated = True
        elif self.battery[self.current_node] <= 0:
            reward -= 100.0
            terminated = True
        elif self.steps >= self.max_steps:
            reward -= 50.0
            truncated = True
        
        # ---- Dynamic Network Simulation ----
        self._simulate_traffic()
        self._simulate_link_failures()
        self._drain_idle_batteries()
        
        return self._get_obs(), reward, terminated, truncated, {}

    def _simulate_traffic(self):
        """Simulate bursty IoT traffic that creates congestion hotspots."""
        for i in range(self.num_nodes):
            if np.random.rand() < self.traffic_intensity:
                self.queue[i] += np.random.uniform(1.0, 4.0)
            # Queues slowly drain (processing)
            self.queue[i] = max(0.0, self.queue[i] - 0.5)
    
    def _simulate_link_failures(self):
        """Randomly fail and recover links to simulate real IoT conditions."""
        # Random link failure
        current_edges = list(self.graph.edges())
        for edge in current_edges:
            if np.random.rand() < self.failure_prob:
                self.graph.remove_edge(*edge)
                # Ensure graph stays connected
                if not nx.is_connected(self.graph):
                    self.graph.add_edge(*edge)  # Revert if it disconnects
        
        # Random link recovery
        for edge in self.original_edges:
            if not self.graph.has_edge(*edge) and np.random.rand() < 0.1:
                self.graph.add_edge(*edge)
    
    def _drain_idle_batteries(self):
        """All nodes slowly lose battery over time (idle drain)."""
        self.battery -= 0.1
        self.battery = np.clip(self.battery, 0.0, 100.0)