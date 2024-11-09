from inference_1 import trade_schedule
import pandas as pd
from CustomBenchmark import CustomBenchmark
from train import order_book

order_book_data = order_book
benchmark = CustomBenchmark(order_book_data)

initial_inventory = 1000

# Convert the DQN trade schedule to DataFrame for compatibility with the benchmark class
dqn_trades_df = pd.DataFrame(trade_schedule, columns=['timestamp', 'shares'])

trade_metrics = benchmark.evaluate_trade_schedule(dqn_trades_df, initial_inventory)
print(trade_metrics)

