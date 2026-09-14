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

def make_env():
    env = IoTRoutingEnv(num_nodes=20)
    env = Monitor(env)
    env = ActionMasker(env, mask_fn)
    return env

def train_model(timesteps=100000):
    # Initialize the environment
    env = make_env()
    eval_env = make_env()
    
    # Callback to evaluate and save best model
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=2000,
                                 deterministic=True, render=False)
                                 
    # Create MaskablePPO model
    model = MaskablePPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")
    
    print(f"Starting training for {timesteps} timesteps...")
    model.learn(total_timesteps=timesteps, callback=eval_callback)
    
    # Save the final model
    model.save("ppo_iot_routing_masked")
    print("Training finished. Final model saved as 'ppo_iot_routing_masked.zip'.")

if __name__ == "__main__":
    train_model(timesteps=80000)
