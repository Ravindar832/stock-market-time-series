import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

def prepare_prophet_data(df, column='Close'):
    """
    Prepare data for Prophet model.
    Converts the index or 'Date' column to datetime and renames to 'ds' and 'y'.
    """
    prophet_df = df[['Date', column]].copy()
    prophet_df['Date'] = pd.to_datetime(prophet_df['Date'])  # ensure datetime
    prophet_df.rename(columns={'Date': 'ds', column: 'y'}, inplace=True)
    return prophet_df

def train_prophet_model(prophet_df):
    """
    Train Prophet model.
    """
    model = Prophet()
    model.fit(prophet_df)
    return model

def forecast_prophet(model, periods=30, freq='D'):
    """
    Forecast future values using Prophet model.
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast

def evaluate_forecast(true_values, predicted_values):
    """
    Evaluate Prophet model using RMSE, MAE, and R2.
    """
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)
    return {"RMSE": round(rmse, 4), "MAE": round(mae, 4), "R2": round(r2, 4)}

# ========== MAIN SCRIPT ==========
if __name__ == "__main__":
    # Load and prepare data
    df = pd.read_csv("../Data/stock_data.csv")
    prophet_df = prepare_prophet_data(df)

    # Train/test split
    split_idx = int(len(prophet_df) * 0.8)
    train_df, test_df = prophet_df.iloc[:split_idx], prophet_df.iloc[split_idx:]

    # Train model
    model = train_prophet_model(train_df)

    # Forecast including test range
    forecast = forecast_prophet(model, periods=len(test_df))

    # Ensure datetime for alignment
    forecast['ds'] = pd.to_datetime(forecast['ds'])
    test_df['ds'] = pd.to_datetime(test_df['ds'])

    # Align forecast with test period
    forecast_test = pd.merge(forecast[['ds','yhat']], test_df, on='ds', how='inner')

    # Evaluate
    metrics = evaluate_forecast(forecast_test['y'], forecast_test['yhat'])
    print("\nProphet Evaluation Metrics:", metrics)

    # Plot train, test, and forecast
    plt.figure(figsize=(12, 5))
    plt.plot(train_df['ds'], train_df['y'], label="Train Data", color="blue")
    plt.plot(test_df['ds'], test_df['y'], label="Actual Test Data", color="green")
    plt.plot(forecast_test['ds'], forecast_test['yhat'], label="Prophet Predictions", color="red", linestyle="--")
    plt.title("Prophet Model: Train/Test Actual vs Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../reports/prophet_forecast.png")
    plt.close()

    # Full model retrain for future forecasting
    full_model = train_prophet_model(prophet_df)
    full_forecast = forecast_prophet(full_model, periods=30)
    future_predictions = full_forecast.tail(30)[['ds', 'yhat']]

    # Plot future 30-day forecast
    plt.figure(figsize=(12, 5))
    plt.plot(prophet_df['ds'], prophet_df['y'], label="Historical Data", color="blue")
    plt.plot(future_predictions['ds'], future_predictions['yhat'], label="30-Day Forecast", color="orange", linestyle="--")
    plt.title("Prophet Model: 30-Day Future Forecast")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("../reports/prophet_future_forecast.png")
    plt.close()

    print("\n🔮 Prophet Next 30-Day Forecast:\n", future_predictions)
