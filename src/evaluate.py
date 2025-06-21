import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_forecast(true_values, predicted_values, model_name=None, streamlit=False, st=None):
    """
    Evaluate forecasting results using RMSE, MAE, and R².

    Args:
        true_values (array-like): Actual target values
        predicted_values (array-like): Forecasted values from the model
        model_name (str): Optional name of the model
        streamlit (bool): If True, print results using Streamlit
        st: The streamlit module (required if streamlit=True)

    Returns:
        dict: A dictionary with RMSE, MAE, and R2 metrics
    """
    rmse = np.sqrt(mean_squared_error(true_values, predicted_values))
    mae = mean_absolute_error(true_values, predicted_values)
    r2 = r2_score(true_values, predicted_values)

    metrics = {
        "RMSE": round(rmse, 4),
        "MAE": round(mae, 4),
        "R2": round(r2, 4)
    }

    if streamlit and st is not None:
        st.write(f"### 📊 {model_name} Evaluation" if model_name else "### 📊 Model Evaluation")
        st.write(f"- RMSE: `{metrics['RMSE']}`")
        st.write(f"- MAE: `{metrics['MAE']}`")
        st.write(f"- R² Score: `{metrics['R2']}`")

    return metrics
