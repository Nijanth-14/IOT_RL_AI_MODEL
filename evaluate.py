import numpy as np
import networkx as nx
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from iot_env import IoTRoutingEnv

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()

def shortest_path_routing(env, start_node, dest_node):
    # Try to find shortest path using NetworkX
    try:
        path = nx.shortest_path(env.graph, source=start_node, target=dest_node)
        if len(path) > 1:
            return path[1] # Return next hop
        return start_node
    except nx.NetworkXNoPath:
        return start_node

def random_routing(env, current_node):
    neighbors = list(env.graph.neighbors(current_node))
    if not neighbors:
        return current_node
    return np.random.choice(neighbors)

def evaluate_model(episodes=10):
    try:
        model = MaskablePPO.load("ppo_iot_routing_masked")
        has_model = True
    except:
        print("Masked PPO model not found, proceeding with baselines only.")
        has_model = False

    base_env = IoTRoutingEnv(num_nodes=20)
    env = ActionMasker(base_env, mask_fn)
    
    methods = ['Random', 'ShortestPath']
    if has_model:
        methods.append('PPO')

    results = {m: {'rewards': [], 'energy': [], 'success': []} for m in methods}

    for method in methods:
        print(f"Evaluating {method}...")
        for ep in range(episodes):
            state, _ = env.reset(seed=42+ep) 
            done = False
            total_reward = 0
            steps = 0
            initial_energy = sum([base_env.state[i][0] for i in range(base_env.num_nodes)])
            
            while not done and steps < 100:
                if method == 'PPO':
                    action_masks = env.action_masks()
                    action, _states = model.predict(state, deterministic=True, action_masks=action_masks)
                elif method == 'ShortestPath':
                    action = shortest_path_routing(base_env, base_env.current_node, base_env.dest_node)
                elif method == 'Random':
                    action = random_routing(base_env, base_env.current_node)
                
                # convert action to native python int if it's numpy scalar
                action = int(action)
                
                state, reward, done, truncated, _ = env.step(action)
                total_reward += reward
                steps += 1
            
            final_energy = sum([base_env.state[i][0] for i in range(base_env.num_nodes)])
            energy_used = initial_energy - final_energy
            
            success = 1 if base_env.current_node == base_env.dest_node else 0
            
            results[method]['rewards'].append(total_reward)
            results[method]['energy'].append(energy_used)
            results[method]['success'].append(success)
            
    print("\nEvaluation Results:")
    for method in methods:
        avg_reward = np.mean(results[method]['rewards'])
        avg_energy = np.mean(results[method]['energy'])
        success_rate = np.mean(results[method]['success']) * 100
        
        print(f"{method}:")
        print(f"  Avg Reward: {avg_reward:.2f}")
        print(f"  Avg Energy Used: {avg_energy:.2f}")
        print(f"  Success Rate: {success_rate:.1f}%")

if __name__ == "__main__":
    evaluate_model(episodes=100)
