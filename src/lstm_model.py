import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler


def prepare_lstm_data(series, window_size=60):
    """
    Prepare data for LSTM input.
    
    Args:
        series (array-like): 1D array of values (e.g., closing prices)
        window_size (int): Number of time steps in each input sequence

    Returns:
        tuple: (X, y, scaler) — reshaped input features, labels, and the fitted scaler
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(np.array(series).reshape(-1, 1))

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i - window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler


def build_lstm_model(input_shape):
    """
    Build and compile the LSTM model.

    Args:
        input_shape (tuple): Shape of input data (timesteps, features)

    Returns:
        Sequential: Compiled LSTM model
    """
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        LSTM(units=50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def forecast_lstm(model, last_sequence, steps, scaler):
    """
    Forecast future values using the trained LSTM model.

    Args:
        model (Sequential): Trained LSTM model
        last_sequence (np.ndarray): Last known window of scaled values (shape: [1, window, 1])
        steps (int): Number of future steps to forecast
        scaler (MinMaxScaler): Fitted scaler used during training

    Returns:
        np.ndarray: Inverse-scaled forecasted values
    """
    predictions = []

    input_seq = last_sequence.copy()
    for _ in range(steps):
        next_pred = model.predict(input_seq, verbose=0)[0][0]
        predictions.append(next_pred)
        input_seq = np.append(input_seq[:, 1:, :], [[[next_pred]]], axis=1)

    return scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
