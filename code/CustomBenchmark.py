#!/usr/bin/env python
import pandas as pd
import numpy as np

class CustomBenchmark:
    def __init__(self, order_book):
        """
        Initializes the CustomBenchmark class with provided order book data.

        Parameters:
        order_book (DataFrame): DataFrame containing order book data including bid and ask prices/sizes.
        """
        self.order_book = order_book

    def evaluate_trade_schedule(self, trade_schedule, initial_inventory):
        """
        Evaluates the trade schedule based on custom benchmarks.

        Parameters:
        trade_schedule (DataFrame): DataFrame containing the trade schedule with columns ['timestamp', 'shares'].
        initial_inventory (int): Initial inventory level of shares to sell.

        Returns:
        DataFrame: Trade metrics for each action in the schedule, including slippage, market impact, and VWAP.
        """
        remaining_inventory = initial_inventory
        trades = []

        for idx, trade in trade_schedule.iterrows():
            timestamp = trade['timestamp']
            shares_to_sell = trade['shares']
            remaining_inventory -= shares_to_sell

            # Retrieve order book data closest to the timestamp
            order_book_row = self.order_book[self.order_book['timestamp'] == timestamp]
            if order_book_row.empty:
                continue

            bid_prices = order_book_row[[f'bid_price_{i}' for i in range(1, 6)]].values.flatten()
            ask_prices = order_book_row[[f'ask_price_{i}' for i in range(1, 6)]].values.flatten()
            bid_sizes = order_book_row[[f'bid_size_{i}' for i in range(1, 6)]].values.flatten()
            ask_sizes = order_book_row[[f'ask_size_{i}' for i in range(1, 6)]].values.flatten()

            # Calculate VWAP for the trade
            vwap_price = self.calculate_vwap(bid_prices, bid_sizes, shares_to_sell)

            # Calculate slippage as the difference between the ideal price and VWAP
            slippage =  (bid_prices[0] - vwap_price) * shares_to_sell  # Assume sell-side for ideal price

            # Market impact estimate
            alpha = 4.439584265535017e-06
            market_impact = alpha * np.sqrt(shares_to_sell)

            # Record trade metrics
            trades.append({
                'timestamp': timestamp,
                'shares': shares_to_sell,
                'vwap_price': vwap_price,
                'slippage': slippage,
                'market_impact': market_impact,
                'remaining_inventory': remaining_inventory
            })

        return pd.DataFrame(trades)

    def calculate_vwap(self, prices, sizes, target_shares):
        """
        Calculate VWAP based on available bid or ask prices and sizes for a given trade.

        Parameters:
        prices (array): Array of prices (e.g., bid or ask).
        sizes (array): Array of sizes corresponding to each price level.
        target_shares (int): Number of shares to calculate VWAP for.

        Returns:
        float: The calculated VWAP price.
        """
        accumulated_volume = 0
        accumulated_value = 0
        for price, size in zip(prices, sizes):
            if accumulated_volume + size >= target_shares:
                accumulated_value += price * (target_shares - accumulated_volume)
                accumulated_volume = target_shares
                break
            else:
                accumulated_value += price * size
                accumulated_volume += size

        if accumulated_volume == 0:
            return 0
        return accumulated_value / accumulated_volume
