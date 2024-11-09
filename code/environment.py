import pandas as pd
import gym
from gym import spaces
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque


# Define the custom AAPL Trading environment
class AAPLTradingEnv(gym.Env):
    def __init__(self, order_book, V=1000, H=390):
        super(AAPLTradingEnv, self).__init__()
        self.order_book = order_book.reset_index(drop=True)
        self.V = V  # Total volume to trade
        self.H = H  # Time horizon

        # Observation space: 5 bid sizes, 5 ask sizes, 5 bid prices, 5 ask prices, normalized time and inventory
        self.observation_space = spaces.Box(low=0, high=1, shape=(22,), dtype=np.float32)

        # Action space: (Hold, Cross Spread, Place in Own Book)
        self.action_space = spaces.Discrete(3)

        self.reset()

    def reset(self):
        self.current_step = 0
        self.remaining_volume = self.V
        return self._get_observation()

    def _get_observation(self):
        row = self.order_book.iloc[self.current_step]

        # Extracting bid/ask sizes and prices
        bid_sizes = row[['bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4', 'bid_size_5']].values
        ask_sizes = row[['ask_size_1', 'ask_size_2', 'ask_size_3', 'ask_size_4', 'ask_size_5']].values
        bid_prices = row[['bid_price_1', 'bid_price_2', 'bid_price_3', 'bid_price_4', 'bid_price_5']].values
        ask_prices = row[['ask_price_1', 'ask_price_2', 'ask_price_3', 'ask_price_4', 'ask_price_5']].values

        normalized_time = self.current_step / self.H
        normalized_inventory = self.remaining_volume / self.V

        # Combine all features into a single observation
        obs = np.concatenate([bid_sizes, ask_sizes, bid_prices, ask_prices, [normalized_time, normalized_inventory]])
        return obs.astype(np.float32)

    def step(self, action):
        row = self.order_book.iloc[self.current_step]
        ask_prices = row[['ask_price_1', 'ask_price_2', 'ask_price_3', 'ask_price_4', 'ask_price_5']].values
        bid_prices = row[['bid_price_1', 'bid_price_2', 'bid_price_3', 'bid_price_4', 'bid_price_5']].values

        reward = 0
        done = False

        if action == 0:  # Hold
            reward = 0
        elif action == 1:  # Cross the spread
            for i in range(5):  # Check all bid sizes
                trade_size = min(self.remaining_volume, row[f'bid_size_{i+1}'])
                if trade_size > 0:
                    reward += (bid_prices[i] - ask_prices[i]) * trade_size  # Reward based on spread
                    self.remaining_volume -= trade_size
                    break  # Only execute one trade per step
        elif action == 2:  # Place in own book
            for i in range(5):  # Check all ask sizes
                trade_size = min(self.remaining_volume, row[f'ask_size_{i+1}'] // 2)  # Place a limit order
                if trade_size > 0:
                    reward += (bid_prices[i] - ask_prices[i]) * trade_size
                    self.remaining_volume -= trade_size
                    break  # Only execute one trade per step

        self.current_step += 1
        if self.current_step >= self.H or self.remaining_volume <= 0:
            done = True

        return self._get_observation(), reward, done, {}

