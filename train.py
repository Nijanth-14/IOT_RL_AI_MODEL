import os
import numpy as np
import gymnasium as gym
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor

from iot_env import IoTRoutingEnv

def mask_fn(env: gym.Env) -> np.ndarray:
    return env.unwrapped.valid_action_mask()

def make_env(seed=42, traffic=0.5, failure=0.05):
    def _init():
        env = IoTRoutingEnv(num_nodes=20, traffic_intensity=traffic, 
                           failure_prob=failure, max_steps=50)
        env = Monitor(env)
        env = ActionMasker(env, mask_fn)
        env.reset(seed=seed)
        return env
    return _init

def train_model(timesteps=200000):
    # Training environment with moderate conditions
    env = make_env(seed=42, traffic=0.5, failure=0.05)()
    # Eval environment with slightly different seed
    eval_env = make_env(seed=123, traffic=0.5, failure=0.05)()
    
    # Callback to evaluate and save best model
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=5000,
                                 deterministic=True, render=False)
    
    # Optimized network architecture
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256, 128], vf=[256, 256, 128])
    )
    
    model = MaskablePPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        tensorboard_log="./tensorboard_logs/",
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=15,
        gamma=0.995,               # Higher discount for longer-horizon planning
        gae_lambda=0.98,
        clip_range=0.2,
        ent_coef=0.01,             # Entropy for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=policy_kwargs,
        seed=42
    )
    
    print(f"Starting training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    
    # Save the final model
    model.save("ppo_iot_routing_v3")
    print("Training finished. Model saved as 'ppo_iot_routing_v3.zip'.")

if __name__ == "__main__":
    train_model()
