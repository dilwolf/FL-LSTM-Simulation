# Federtaed LSTM Model Training on the custom dataset

This repository demonstrates a simulation of the training process of an LSTM model using Federated Learning (FL) on a custom dataset. 

## Overview

The following figure shows an overall comparison of the convergence of the LSTM model during the training process in terms of federated and centralized learning on the same dataset as presented in Table 3 of our paper. We trained the LSTM model for 200 rounds and evaluated the training results using the RMSE metric. Specifically, we first recorded the required communication rounds for both the federated and centralized training results. 

We also compared the convergence of the LSTM model across different numbers of clients (K) (e.g., K = 1 means centralized training) in a federated manner, as shown in Figure 5. We assessed the accuracy of our LSTM model in terms of the RMSE. The convergence results indicate that the centralized-based LSTM model achieved the lowest RMSE value of 0.79, followed by the FL-based LSTM models with an increasing number of clients. One significant reason for this is that the dataset was distributed among clients in the FL-based LSTM during the training process, which influenced model performance.

While centralized training can produce better outcomes in these circumstances, it is crucial to acknowledge the unparalleled advantages of FL-based LSTM training regarding privacy and security. FL enables collaboration among parties while safeguarding data integrity, making it an ideal choice where maintaining ownership and data privacy are important. Considering these advantages, our research recommends adopting FL-based LSTM training as the approach for privacy, even though there were some performance differences compared to centralized training.

## Prerequisites

- Python 3.7 or later
- Required Python packages (listed in `requirements.txt`)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/dilwolf/FL-LSTM-Simulation.git
   cd FL-LSTM-Simulation
2. Create and activate conda environment:
   ```bash
   conda create -n fl-lstm python=3.10
   conda activate fl-lstm

3. Install the required packages:
   ```bash
   pip install -r requirements

# Usage
## Centralized Training

To run centralized training:
  ```bash
  python evaluate.py --model models/centr_model.python


# Results
The results of the training process, including the convergence graphs and RMSE values, can be found in the results/ directory.



