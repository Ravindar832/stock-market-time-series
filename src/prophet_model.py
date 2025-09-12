import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np


def prepare_prophet_data(df, column='Close'):
    """
    Prepare data for Prophet model.

    

    Returns:
        pd.DataFrame: Reformatted DataFrame with 'ds' and 'y' columns
    """
    prophet_df = df[[column]].reset_index()
    prophet_df.columns = ['ds', 'y']
    return prophet_df


def train_prophet_model(prophet_df):
    """
    Train Prophet model.

    Args:
        prophet_df (pd.DataFrame): DataFrame with 'ds' and 'y'

    Returns:
        model: Fitted Prophet model
    """
    model = Prophet()
    model.fit(prophet_df)
    return model


def forecast_prophet(model, periods=30, freq='D'):
    """
    Forecast future values using Prophet model.

    Args:
        model: Trained Prophet model
        periods (int): Number of future time steps to forecast
        freq (str): Frequency of forecast (e.g., 'D' for daily)

    Returns:
        pd.DataFrame: Prophet forecast DataFrame with columns ['ds', 'yhat', 'yhat_lower', 'yhat_upper']
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast


def evaluate_forecast(true_values, predicted_values):
    """
    Evaluate Prophet model using RMSE, MAE, and R2.

    Args:
        true_values (array-like): Actual values
        predicted_values (array-like): Predicted values

    Returns:
        dict: Dictionary of evaluation metrics
    """
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    return {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4)
    }
# ========== MAIN SCRIPT ==========
if __name__ == "__main__":
    
    df = pd.read_csv("../Data/stock_data.csv")
    print(df.head())
    
     # Prepare data for Prophet: rename columns to 'ds' and 'y'
    prophet_df = df[['Date', 'Close']].copy()
    prophet_df.rename(columns={'Date': 'ds', 'Close': 'y'}, inplace=True)

    # Split train/test (80/20)
    split_idx = int(len(prophet_df) * 0.8)
    train_df, test_df = prophet_df.iloc[:split_idx], prophet_df.iloc[split_idx:]

    # Train Prophet
    forecast, model = train_prophet_model(train_df, steps=len(test_df))

    # Evaluate on test set
    # Align forecast with test period
    forecast_test = forecast.set_index('ds').loc[test_df['ds']]
    metrics = evaluate_forecast(test_df['y'], forecast_test['yhat'])
    print("\n Prophet Evaluation Metrics:", metrics)

    # Forecast future 30 steps
    full_forecast, _ = train_prophet_model(prophet_df, steps=30)
    future_predictions = full_forecast.tail(30)[['ds', 'yhat']]
    print("\n🔮 Prophet Next 30-Day Forecast:\n", future_predictions)
