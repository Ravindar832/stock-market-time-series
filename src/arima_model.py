import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import os  

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
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from statsmodels.tsa.arima.model import ARIMA

# Force CPU & suppress TF logs
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


# ========== ARIMA Utilities ==========
def train_arima_model(series, order=(5, 1, 0), steps=30):
    model = ARIMA(series, order=order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=steps)
    return forecast, model_fit


def evaluate_forecast(true_values, predicted_values):
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}


# ========== MAIN SCRIPT ==========
if __name__ == "__main__":
    
    df = pd.read_csv("../Data/stock_data.csv")
    print(df.head())

    # Select 'Close' column
    series = df['Close']

    # Split train/test (80/20)
    split_idx = int(len(series) * 0.8)
    train, test = series[:split_idx], series[split_idx:]

    # Train ARIMA
    forecast, model_fit = train_arima_model(train, order=(5, 1, 0), steps=len(test))

    # Evaluate
    metrics = evaluate_forecast(test.values, forecast.values)
    print("\n ARIMA Evaluation Metrics:", metrics)

    # Forecast future 30 steps ahead
    full_forecast, _ = train_arima_model(series, order=(5, 1, 0), steps=30)
    print("\n ARIMA Next 30-Day Forecast:\n", full_forecast.values)
    
    plt.figure(figsize=(12,5))
    plt.plot(series.index, series, label="Actual", color="blue")
    plt.plot(test.index, forecast, label="ARIMA Forecast", color="red")
    plt.title("ARIMA Model Forecast")
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.legend()
    plt.savefig("arima_forecast.png")
    plt.savefig("../reports/arima_forecast.png")  
    plt.close()