# src/train.py
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import os
import sys

# Engineering Standard: Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path: str) -> pd.DataFrame:
    """Loads the validated dataset."""
    logger.info(f"Loading training data from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"Data file missing: {file_path}")
        sys.exit(1)
    return pd.read_csv(file_path)

def train_and_track():
    """Trains the XGBoost model and logs all artifacts to MLflow."""
    # 1. Connect to the local MLflow server
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Demand_Forecasting_Baseline")

    # 2. Load the data
    data_path = os.path.join("data", "processed", "baseline_train.csv")
    df = load_data(data_path)

    # Define our features and target variable
    feature_cols = ['store', 'item', 'day_of_week', 'month', 'year', 'rolling_7d_sales']
    X = df[feature_cols]
    y = df['sales']

    # 3. Define Hyperparameters
    params = {
        "n_estimators": 100,
        "max_depth": 5,
        "learning_rate": 0.1,
        "objective": "reg:squarederror"
    }

    logger.info("Starting MLflow tracking run...")
    
    # 4. Execute the training run within MLflow's context manager
    with mlflow.start_run():
        # Log the hyperparameters
        mlflow.log_params(params)

        # Train the model
        logger.info("Training XGBoost model...")
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)

        # Evaluate the model on the training set (in a real scenario, you'd use a holdout set)
        logger.info("Evaluating model performance...")
        predictions = model.predict(X)
        mae = mean_absolute_error(y, predictions)
        rmse = mean_squared_error(y, predictions, squared=False)

        # Log the metrics
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        logger.info(f"Metrics - MAE: {mae:.4f} | RMSE: {rmse:.4f}")

        # Log the actual model binary (the artifact)
        mlflow.xgboost.log_model(model, "xgboost-model")
        
        logger.info("Model and artifacts successfully logged to MLflow.")

if __name__ == "__main__":
    train_and_track()