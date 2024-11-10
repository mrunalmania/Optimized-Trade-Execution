# Optimized Trade Execution Using Reinforcement Learning

**Author:** Mrunal Ashwinbhai Mania  
**Institution:** Arizona State University  
**Email:** mmania1@asu.edu  

## Overview
This project presents a Deep Q-Network (DQN)-based reinforcement learning (RL) framework for optimizing trade execution in financial markets. The agent leverages limit order book data to determine optimal trading actions, aiming to minimize market impact and slippage while adhering to execution benchmarks like Volume Weighted Average Price (VWAP). This approach is especially useful for large institutional investors looking to execute large orders with minimal cost inefficiency.

## Project Objectives
- Minimize the impact of large orders on market prices.
- Execute trades close to the VWAP to optimize cost-efficiency.
- Learn an optimal strategy from historical order book data for effective sell-side trade scheduling.

## Features
- **Custom Trading Environment:** Simulates real-world trading scenarios with actions to hold or sell in varying quantities.
- **Deep Q-Learning (DQN) Model:** Uses a discrete action space and approximates Q-values for efficient decision-making.
- **Evaluation Metrics:** Includes VWAP adherence, slippage, market impact, and inventory depletion.
- **Scalable Deployment on AWS SageMaker:** The model is designed for real-time deployment, allowing live trading decisions with current market data.

## Methodology
1. **Data Preprocessing**: Historical AAPL order book data, including bid/ask prices and volumes, are transformed to represent the model’s state.
2. **Environment Design**: Defines the state and action spaces and implements a reward function to guide the model toward optimal execution.
3. **DQN Model Architecture**: Neural network architecture captures market patterns to predict the expected rewards for each action.
4. **Training and Fine-Tuning**: Experience replay and a target network help stabilize learning, while hyperparameters and the reward function are tuned for optimal performance.

## Model Architecture
- **State Space**: Features derived from real-time bid/ask prices, sizes, and order book depth.
- **Action Space**: Discrete choices of holding or selling varying quantities of shares.
- **Reward Function**: Combines immediate rewards (cost minimization) with long-term adherence to VWAP.

## Evaluation
Key metrics are used to measure the model’s performance:
- **VWAP Adherence**: Alignment with VWAP to reduce cost inefficiency.
- **Slippage**: Difference between expected and actual trade prices.
- **Market Impact**: Effect of trade execution on the overall market.
- **Remaining Inventory**: Effectiveness in executing the entire order within the specified timeframe.

## Deployment
The model is prepared for deployment as an AWS SageMaker real-time endpoint, providing live trading predictions. The deployment pipeline integrates Docker, SageMaker, and other dependencies to support scalable, real-time inference.

## Getting Started
### Prerequisites
- Python 3.8+
- PyTorch, Pandas, NumPy, Ray (for distributed RL training)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/optimized-trade-execution.git
   cd optimized-trade-execution
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Model
1. **Training**: Train the DQN agent using historical order book data.
   ```bash
   python train.py
   ```
2. **Inference**: Generate a trade schedule based on the trained model.
   ```bash
   python inference.py
   ```

### AWS Deployment
The project includes a deployment script using AWS SageMaker and `boto3`. Update the deployment parameters in `environment.py`, then run the script to deploy the model.

## Future Work
- **Adaptive Reward Function**: To account for real-time volatility and liquidity.
- **Extended Dataset**: Testing on diverse stocks and conditions to improve generalization.
- **Live Deployment**: Finalize AWS SageMaker deployment for real-time trading execution.

## References
- Nevmyvaka, Y., Feng, Y., Kearns, M. (2005). Reinforcement Learning for Optimized Trade Execution.
- Sutton, R. S., Barto, A. G. (2018). Reinforcement Learning: An Introduction.
- Ray RLLib Documentation, Ray (2024).

## License
[MIT License](https://en.wikipedia.org/wiki/MIT_License)
