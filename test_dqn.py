from preprocess_btc_data import *
from backtester import Backtester
from dqn import DQN
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

def setup_gpus():
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if physical_devices:
        try:
            for gpu in physical_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)

def test_dqn(model_path):
    setup_gpus()

    # Load historical Bitcoin price data
    data = pd.read_csv('datasets/btc_price_data.csv') # Assuming a CSV with a 'Price' column
    data = preprocess_for_testing(data)
    backtester = Backtester(data)
    backtester.previous_net_worth = backtester.initial_balance  # Set the initial previous_net_worth

    # Initialize the DQN strategy
    window_size = 10
    action_space = 20
    learning_rate = 0.001
    dqn_strategy = DQN(input_dim=window_size, action_space=action_space, learning_rate=learning_rate)

    # Load the trained model
    dqn_strategy.load_model(model_path)

    # Backtest using the loaded model on the testing data
    # net_worths = backtester.backtest(dqn_strategy, window_size, batch_size=32)
    net_worths = backtester.backtest(dqn_strategy, window_size)
    # Plot the net worth over time
    pd.Series(net_worths).plot(title="Net Worth Over Time: DQN")
    plt.xlabel('Time (Elapsed Hours)')
    plt.ylabel('Net Worth ($)')
    plt.savefig('DQN_evaluation_net_worth_over_time.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    test_dqn("dqn_model_after_epoch_2.h5") # Make sure to put the correct model file here.
