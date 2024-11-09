#!/usr/bin/env python
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Define the DQN neural network
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# DQN agent class
class DQNAgent:
    def __init__(self, input_dim, output_dim):
        self.memory = deque(maxlen=2000)
        self.gamma = 0.99  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.85
        self.model = DQN(input_dim, output_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = nn.MSELoss()

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(3)  # Explore
        state = torch.FloatTensor(state).unsqueeze(0)
        act_values = self.model(state)
        return torch.argmax(act_values[0]).item()  # Exploit

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(torch.FloatTensor(next_state).unsqueeze(0))).item()
            target_f = self.model(torch.FloatTensor(state).unsqueeze(0))
            target_f[0][action] = target
            self.optimizer.zero_grad()
            loss = self.loss_fn(target_f, torch.FloatTensor(target_f).unsqueeze(0))
            loss.backward()
            self.optimizer.step()

# Training the DQN agent
def train_agent(env, episodes):
    agent = DQNAgent(input_dim=22, output_dim=3)  # Adjust based on observation space
    for e in range(episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        agent.replay(32)  # Replay experiences

        # Epsilon decay
        if agent.epsilon > agent.epsilon_min:
            agent.epsilon *= agent.epsilon_decay

        print(f"Episode: {e+1}/{episodes}, Total Reward: {total_reward}")

    return agent