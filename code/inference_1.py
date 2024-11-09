#!/usr/bin/env python

import pandas as pd
import gym
from gym import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

from train import trained_agent, env

# Generate trade schedule
def generate_trade_schedule(agent, env, total_volume):
    current_step = 0
    remaining_volume = total_volume
    trade_schedule = []

    while current_step < env.H and remaining_volume > 0:
        obs = env.reset() if current_step == 0 else obs
        action = agent.act(obs)

        if action == 0:  # Hold
            pass
        elif action == 1:  # Cross the spread
            for i in range(5):
                trade_size = min(remaining_volume, env.order_book.iloc[current_step][f'bid_size_{i+1}'])
                if trade_size > 0:
                    trade_schedule.append((env.order_book.iloc[current_step]['timestamp'], int(trade_size)))
                    remaining_volume -= trade_size
                    break  # Only execute one trade per step
        elif action == 2:  # Place in own book
            for i in range(5):
                trade_size = min(remaining_volume, env.order_book.iloc[current_step][f'ask_size_{i+1}'] // 2)
                if trade_size > 0:
                    trade_schedule.append((env.order_book.iloc[current_step]['timestamp'], int(trade_size)))
                    remaining_volume -= trade_size
                    break  # Only execute one trade per step

        current_step += 1
        obs, _, _, _ = env.step(action)

    return trade_schedule

# Create a trade schedule
trade_schedule = generate_trade_schedule(trained_agent, env, total_volume=1000)

# Print the trade schedule
for timestamp, size in trade_schedule:
    print(f"Timestamp: {timestamp}, Shares: {size}")
