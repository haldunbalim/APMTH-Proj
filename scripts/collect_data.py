import argparse
import os
import pickle

import numpy as np
from apmth import *
import gymnasium as gym
from grid2op.Agent import RandomAgent  # Changed from DoNothingAgent
import datetime
from tqdm import tqdm

def collect_data(env, num_episodes=1_000):
    data = []

    for _ in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()  # Random action
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            data.append((obs, action, next_obs, reward, terminated, truncated))
            obs = next_obs

    os.makedirs('data', exist_ok=True)
    fname = f"random_data_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
    with open(os.path.join('data', fname), "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    env = gym.make('l2rpn_case14_sandbox_train-v0')
    collect_data(env)
