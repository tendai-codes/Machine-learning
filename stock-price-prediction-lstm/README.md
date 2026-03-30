# Stock Price Prediction Using Long Short-Term Memory (LSTM)

## Problem

This project analysed historical stock price time-series data to model temporal patterns and support prediction of future stock price behaviour using a Long Short-Term Memory (LSTM) neural network.

## Objective

- Prepare historical stock price data for sequence-based time-series modelling
- Construct an LSTM neural network for stock price prediction
- Train the model on historical price sequences
- Compare predicted values with observed stock price behaviour

## Approach

- Imported historical stock price data and selected relevant time-series features
- Normalised price values to support neural network training stability
- Generated sequential training datasets using sliding time windows
- Trained an LSTM model and produced predictions on evaluation data

## Key Findings

- The LSTM model generated predicted stock price sequences from historical input windows
- Normalisation supported stable training of the time-series neural network
- Sequential input construction enabled modelling of temporal dependencies in price data
- Visual comparison plots showed alignment between predicted and observed price trends over the evaluation period

## Notbook
- `Stock Price Predictions using Long Short Term Memory Model.ipynb`

