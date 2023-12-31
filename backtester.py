# backtester.py
import pandas as pd
from tqdm import tqdm
import matplotlib as plt
import numpy as np

class Backtester:
    def __init__(self, data, initial_balance=10000):
        self.data = data
        self.initial_balance = initial_balance
        self.action_counter = {i: 0 for i in range(20)}  # Assuming 20 different actions
        self.reset()

    def reset(self):
        self.balance = self.initial_balance
        self.held_bitcoins = 0
        self.net_worth = self.initial_balance

    def step(self, action, current_price):
        """
        Execute a trading action with transaction costs:
        0-9 -> Buy in 10% increments
        10 -> Hold
        11-19 -> Sell in 10% increments
        Transaction cost is 1% of the traded amount.
        """
        # Increment the action counter
        self.action_counter[action] += 1

        transaction_cost_rate = 0.01  # 1% transaction cost

        if action < 10:  # Buy
            buy_fraction = (action + 1) * 0.1  # 10% to 100%
            investment = self.balance * buy_fraction
            transaction_cost = investment * transaction_cost_rate
            investment_after_cost = investment - transaction_cost
            self.held_bitcoins += investment_after_cost / current_price
            self.balance -= investment  # Subtract the original investment from balance

        elif action > 10:  # Sell
            sell_fraction = (action - 10) * 0.1  # 10% to 90%
            bitcoins_to_sell = self.held_bitcoins * sell_fraction
            revenue = bitcoins_to_sell * current_price
            transaction_cost = revenue * transaction_cost_rate
            revenue_after_cost = revenue - transaction_cost
            self.balance += revenue_after_cost
            self.held_bitcoins -= bitcoins_to_sell
        
        # Update the net worth
        self.net_worth = self.balance + self.held_bitcoins * current_price
        return self.net_worth

    def get_state(self, i, window_size=10):
        if i - window_size + 1 < 0:
            # If not enough past data, pad with zeros
            pad_size = abs(i - window_size + 1)
            pad = np.zeros(pad_size)
            window_data = np.concatenate((pad, self.data.iloc[:i+1]['Price'].values))
        else:
            window_data = self.data.iloc[i-window_size+1:i+1]['Price'].values
            
        # Normalize data to be in range [0, 1] (you can use other normalization techniques)
        normalized_data = (window_data - np.min(window_data)) / (np.max(window_data) - np.min(window_data) + 1e-10)
        
        return normalized_data.reshape(1, window_size)

    # def backtest(self, strategy, window_size, batch_size=32):
    #     self.reset()
    #     net_worths = []
        
    #     # Adjust the range to avoid fetching states beyond data length
    #     for i in tqdm(range(len(self.data) - window_size), desc="Backtesting", ncols=100):  
    #         state = self.get_state(i)  # Get the current state
    #         action = strategy.act(state)
    #         price = self.data.iloc[i]['Price']
    #         net_worth = self.step(action, price)
            
    #         # Ensure next_state is fetched only if we are not near the data's end
    #         if i < len(self.data) - window_size - 1:  
    #             next_state = self.get_state(i+1)
    #             reward = net_worth - net_worths[-1] if net_worths else net_worth
    #             strategy.remember(state, action, reward, next_state, i == len(self.data)-1-window_size)
    #             strategy.replay(batch_size)
                
    #         net_worths.append(net_worth)
            
    #     return net_worths
    def backtest(self, strategy, window_size):
        self.reset()
        net_worths = []

        for i in range(window_size, len(self.data)):  
            state = self.get_state(i, window_size)
            action = strategy.act(state)
            price = self.data.iloc[i]['Price']
            net_worth = self.step(action, price)
            net_worths.append(net_worth)

        # Return both net worths and action counter
        return net_worths, self.action_counter



def sample_strategy(data):
    """
    A sample strategy to demonstrate backtesting.
    Buys if the current price is higher than the last, else sells.
    """
    if len(data) < 2:
        return 1 # Hold if we don't have enough data
    
    if data.iloc[-1]['Price'] > data.iloc[-2]['Price']:
        return 0 # Buy
    else:
        return 2 # Sell