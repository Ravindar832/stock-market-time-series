import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

def train_arima_model(series, order=(5, 1, 0), steps=30):
    """
    Trains an ARIMA model and forecasts future values.
    
    Args:
        series (pd.Series): Time series data (e.g., closing prices)
        order (tuple): ARIMA order (p,d,q)
        steps (int): Number of steps to forecast
    
    Returns:
        forecast (np.ndarray): Forecasted values
        model_fit: Trained ARIMA model object
    """
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast, model_fit


def evaluate_forecast(true_values, predicted_values):
    """
    Returns RMSE, MAE, and R² for forecast evaluation.
    
    Args:
        true_values (array-like): Actual values
        predicted_values (array-like): Predicted values
    
    Returns:
        dict: RMSE, MAE, R² metrics
    """
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4)
    }
