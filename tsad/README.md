# Time Series Anomaly detection based on Deep Learning

**TSAD** is a Python module created for solving Anomaly Detection problems with time series data. The module is based on deep learning techniques.

The main meaning of this module are:

1. Forecast a multivariate Time Series (TS) one point ahead (Also works for univariate TS)
2. Compute residuals between forecast and true values
3. Apply analysis of residuals (default is Hoteling Statics)
4. Plot and return anomalies

This module allows forecast multi-step ahead both multivariate and univariate time series also.

As forecasting algorithms were implemented or will be implemented:

- A simple one-layer LSTM network (LSTM) 
- A two-layer LSTM network (DeepLSTM) 
- A bi-directional LSTM network (BLSTM) 
- LSTM encoder-decoder (EncDec-AD) 
- LSTM autoencoder (LSTM-AE);
- Convolutional LSTM network (ConvLSTM) 
- Convolutional Bi-directional LSTM network (CBLSTM) 
- Multi-Scale Convolutional Recurrent Encoder-Decoder (MSCRED) 

Actually, the possibility of the module allows you to use any own forecasting algorithm, computer of residuals, or evaluator of residuals. 

#### Documentation

https://tsad.readthedocs.io/
