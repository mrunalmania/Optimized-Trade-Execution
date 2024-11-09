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
from environment import AAPLTradingEnv
from model import train_agent

# Load the AAPL order book CSV
file_path = "AAPL_Quotes_Data.csv"  # Replace with your actual file path
order_book = pd.read_csv(file_path)

# Initialize the environment and train the agent
env = AAPLTradingEnv(order_book=order_book, V=1000, H=390)
trained_agent = train_agent(env, episodes=100)

import torch

torch.save(trained_agent.model.state_dict(), "dqn_model.pth")