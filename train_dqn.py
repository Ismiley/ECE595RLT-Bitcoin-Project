import pandas as pd
import numpy as np
import os
import csv
import tensorflow as tf
from preprocess_btc_data import *
from backtester import Backtester
from dqn import DQN
from tqdm import tqdm

def setup_gpus():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def initialize_csv(file_name, fieldnames):
    if not os.path.exists(file_name):
        with open(file_name, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fieldnames)

def train_dqn():
    setup_gpus()

    # Load historical Bitcoin price data
    data = pd.read_csv('datasets/btc_price_data.csv')
    data = preprocess_for_training(data)
    backtester = Backtester(data)
    backtester.previous_net_worth = backtester.initial_balance  # Set the initial previous_net_worth

    # Initialize the DQN strategy
    window_size = 10
    action_space = 20
    learning_rate = 0.001
    dqn_strategy = DQN(input_dim=window_size, action_space=action_space, learning_rate=learning_rate)

    # Initialize metrics storage and CSV file
    metrics_filename = 'dqn_training_metrics.csv'
    initialize_csv(metrics_filename, ['Epoch', 'Average Reward', 'Loss', 'Epsilon'])

    num_epochs = 10
    batch_size = 32  # Define batch size for training
    steps_until_replay = 0  # Counter for steps until replay

    for epoch in range(num_epochs):
        print(f"Training Epoch {epoch + 1}/{num_epochs}...")

        epoch_rewards = []
        epoch_losses = []

        # Backtest using the DQN strategy
        for i in tqdm(range(window_size, len(data)), desc="Backtesting", ncols=100):
            state = backtester.get_state(i, window_size)
            action = dqn_strategy.act(state)
            price = data.iloc[i]['Price']
            net_worth = backtester.step(action, price)
            
            next_state = backtester.get_state(i + 1) if i < len(data) - 1 else None
            done = i == len(data) - 1
            reward = net_worth - backtester.previous_net_worth
            dqn_strategy.remember(state, action, reward, next_state, done)
            
            steps_until_replay += 1
            if steps_until_replay >= batch_size or done:  # Check if it's time to replay
                loss = dqn_strategy.replay(batch_size)
                steps_until_replay = 0  # Reset steps counter after replay
                if loss is not None:  # Only append if loss is a number
                    epoch_losses.append(loss)
            
            epoch_rewards.append(reward)

        # Compute average metrics
        average_reward = np.mean(epoch_rewards) if epoch_rewards else 0
        average_loss = np.mean(epoch_losses) if epoch_losses else 0
        current_epsilon = dqn_strategy.epsilon

        # Store metrics
        with open(metrics_filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, average_reward, average_loss, current_epsilon])

        # Save the trained model after each epoch
        dqn_strategy.save_model(f"dqn_model_after_epoch_{epoch + 1}.h5")

if __name__ == "__main__":
    train_dqn()
