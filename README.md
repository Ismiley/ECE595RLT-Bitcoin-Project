# ECE595RLT-Bitcoin-Project
## Introduction
This repository contains the code for the project "Maximizing Long-Term Profits in Bitcoin Trading Using Reinforcement Learning." The project focuses on developing a trading agent using a Deep Q-Network (DQN) model to navigate the Bitcoin market effectively.

## Key Features
- Implementation of a DQN model for Bitcoin trading.
- Analysis of high-frequency Bitcoin price data.
- Strategies for long-term profitability in cryptocurrency trading.

## Getting Started
To get started with this project, you will need to clone the repository and install the necessary dependencies.

### Cloning the Repository
```bash
git clone https://github.com/Ismiley/ECE595RLT-Bitcoin-Project.git
cd ECE595RLT-Bitcoin-Project
```
## Installing Dependencies
This project requires several dependencies to run. You can install them using the following command:
```bash
pip install -r requirements.txt
```

## Usage
# Steps to download the dataset:
1) Go to the website: https://www.kaggle.com/datasets/mczielinski/bitcoin-historical-data
2) Click download: This is a 105MB file.


# Steps to train model:
1) In backtester.py , define your transaction cost rate for the backtester by setting a value for the 'transaction_cost_rate' variable.
2)
```bash
python3 train_dqn.py
```

# Steps to test model:
1) In backtester.py , define your transaction cost rate for the backtester by setting a value for the 'transaction_cost_rate' variable.
2) In test_dqn.py , define the model to upload for testing.
3) 
```bash
python3 test_dqn.py
```
