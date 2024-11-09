#!/usr/bin/env python

# inference.py
import json
import torch
import numpy as np
from model import DQNAgent  # Your custom model class
from environment import CustomTradingEnv  # Your custom environment class

# Load the model
def model_fn(model_dir):
    model = DQNAgent()
    model.load_state_dict(torch.load(f"{model_dir}/dqn_model.pth"))
    model.eval()
    return model

# Define how to handle inputs and make predictions
def input_fn(request_body, request_content_type):
    if request_content_type == 'application/json':
        request = json.loads(request_body)
        return request['ticker'], request['shares'], request['time_horizon']
    else:
        raise ValueError("Unsupported content type: {}".format(request_content_type))

def predict_fn(input_data, model):
    ticker, shares, time_horizon = input_data
    # Initialize the environment and make predictions
    env = CustomTradingEnv()
    state = env.reset()  # Reset the environment to start a new trading session

    trade_schedule = []
    remaining_shares = shares

    while remaining_shares > 0:
        action = model(state)
        next_state, reward, done, _ = env.step(action)
        trade_schedule.append({
            "timestamp": env.current_timestamp(),
            "shares": action
        })
        remaining_shares -= action
        state = next_state
        if done:
            break

    return trade_schedule

def output_fn(prediction, content_type):
    return json.dumps(prediction)
