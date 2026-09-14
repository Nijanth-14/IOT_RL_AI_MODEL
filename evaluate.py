import numpy as np
import networkx as nx
import gymnasium as gym
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from iot_env import IoTRoutingEnv

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()

# ============================================================
# Baseline Algorithms
# ============================================================

def shortest_path_routing(env, current_node, dest_node):
    """Standard Dijkstra shortest path - ignores energy and congestion."""
    try:
        path = nx.shortest_path(env.graph, source=current_node, target=dest_node)
        if len(path) > 1:
            return path[1]
        return current_node
    except nx.NetworkXNoPath:
        # If no path exists (due to link failure), pick random neighbor
        neighbors = list(env.graph.neighbors(current_node))
        return np.random.choice(neighbors) if neighbors else current_node

def energy_aware_shortest_path(env, current_node, dest_node):
    """Dijkstra weighted by inverse battery - avoids low-energy nodes but ignores congestion."""
    try:
        # Weight edges by inverse battery of destination node
        for u, v in env.graph.edges():
            w = max(0.1, 100.0 - env.battery[v]) + max(0.1, 100.0 - env.battery[u])
            env.graph[u][v]['weight'] = w
        path = nx.shortest_path(env.graph, source=current_node, target=dest_node, weight='weight')
        if len(path) > 1:
            return path[1]
        return current_node
    except nx.NetworkXNoPath:
        neighbors = list(env.graph.neighbors(current_node))
        return np.random.choice(neighbors) if neighbors else current_node

def random_routing(env, current_node):
    """Random walk - picks a random valid neighbor."""
    neighbors = list(env.graph.neighbors(current_node))
    if not neighbors:
        return current_node
    return np.random.choice(neighbors)

# ============================================================
# Multi-Scenario Evaluation
# ============================================================

SCENARIOS = {
    'Normal':       {'traffic': 0.3, 'failure': 0.02},
    'Congested':    {'traffic': 0.7, 'failure': 0.02},
    'Unstable':     {'traffic': 0.3, 'failure': 0.10},
    'Extreme':      {'traffic': 0.8, 'failure': 0.10},
}

def run_episode(base_env, env, method, model, seed):
    """Run a single episode and return metrics."""
    state, _ = env.reset(seed=seed)
    done = False
    total_reward = 0
    steps = 0
    initial_energy = float(np.sum(base_env.battery))
    
    while not done and steps < 100:
        if method == 'PPO':
            action_masks = env.action_masks()
            action, _ = model.predict(state, deterministic=True, action_masks=action_masks)
        elif method == 'ShortestPath':
            action = shortest_path_routing(base_env, base_env.current_node, base_env.dest_node)
        elif method == 'EnergyAwareSP':
            action = energy_aware_shortest_path(base_env, base_env.current_node, base_env.dest_node)
        elif method == 'Random':
            action = random_routing(base_env, base_env.current_node)
        
        action = int(action)
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        total_reward += reward
        steps += 1
    
    final_energy = float(np.sum(base_env.battery))
    energy_used = initial_energy - final_energy
    success = 1 if base_env.current_node == base_env.dest_node else 0
    min_battery = float(np.min(base_env.battery))
    
    return total_reward, energy_used, success, steps, min_battery

def evaluate_all(episodes=100):
    # Load model
    try:
        model = MaskablePPO.load("ppo_iot_routing_v3")
        print("Loaded ppo_iot_routing_v3 model.")
        has_model = True
    except:
        try:
            model = MaskablePPO.load("ppo_iot_routing_masked")
            print("Loaded ppo_iot_routing_masked model (fallback).")
            has_model = True
        except:
            model = None
            has_model = False
            print("No model found. Evaluating baselines only.")

    methods = ['Random', 'ShortestPath', 'EnergyAwareSP']
    if has_model:
        methods.append('PPO')
    
    all_results = {}
    
    for scenario_name, params in SCENARIOS.items():
        print(f"\n{'='*60}")
        print(f"Scenario: {scenario_name} (traffic={params['traffic']}, failure={params['failure']})")
        print(f"{'='*60}")
        
        results = {m: {'rewards': [], 'energy': [], 'success': [], 
                       'steps': [], 'min_battery': []} for m in methods}
        
        for method in methods:
            for ep in range(episodes):
                base_env = IoTRoutingEnv(
                    num_nodes=20,
                    traffic_intensity=params['traffic'],
                    failure_prob=params['failure'],
                    max_steps=50
                )
                env = ActionMasker(base_env, mask_fn)
                
                r, e, s, st, mb = run_episode(base_env, env, method, model, seed=42+ep)
                results[method]['rewards'].append(r)
                results[method]['energy'].append(e)
                results[method]['success'].append(s)
                results[method]['steps'].append(st)
                results[method]['min_battery'].append(mb)
        
        all_results[scenario_name] = results
        
        # Print results for this scenario
        print(f"\n{'Method':<18} {'Avg Reward':>12} {'Avg Energy':>12} {'Success%':>10} {'Avg Steps':>10} {'Min Batt':>10}")
        print("-" * 74)
        for method in methods:
            avg_r = np.mean(results[method]['rewards'])
            avg_e = np.mean(results[method]['energy'])
            sr = np.mean(results[method]['success']) * 100
            avg_s = np.mean(results[method]['steps'])
            avg_mb = np.mean(results[method]['min_battery'])
            print(f"{method:<18} {avg_r:>12.2f} {avg_e:>12.2f} {sr:>9.1f}% {avg_s:>10.2f} {avg_mb:>10.2f}")
    
    # Generate comparison plots
    generate_plots(all_results, methods)
    return all_results

def generate_plots(all_results, methods):
    """Generate comparison bar charts for all scenarios."""
    scenarios = list(all_results.keys())
    metrics = ['rewards', 'energy', 'success', 'steps']
    metric_labels = ['Avg Reward', 'Avg Energy Used', 'Success Rate (%)', 'Avg Steps']
    
    colors = {'Random': '#e74c3c', 'ShortestPath': '#3498db', 
              'EnergyAwareSP': '#f39c12', 'PPO': '#2ecc71'}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('IoT Routing: RL Agent vs Baselines Across Network Conditions', 
                 fontsize=16, fontweight='bold')
    
    for idx, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[idx // 2][idx % 2]
        
        x = np.arange(len(scenarios))
        width = 0.18
        
        for i, method in enumerate(methods):
            values = []
            for scenario in scenarios:
                data = all_results[scenario][method][metric]
                if metric == 'success':
                    values.append(np.mean(data) * 100)
                else:
                    values.append(np.mean(data))
            
            offset = (i - len(methods)/2 + 0.5) * width
            bars = ax.bar(x + offset, values, width, label=method, 
                         color=colors.get(method, '#95a5a6'), alpha=0.85,
                         edgecolor='white', linewidth=0.5)
        
        ax.set_xlabel('Network Scenario')
        ax.set_ylabel(label)
        ax.set_title(label, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(scenarios)
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=150, bbox_inches='tight')
    print("\nPlot saved to evaluation_results.png")

if __name__ == "__main__":
    evaluate_all(episodes=100)
