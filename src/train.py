import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import root_mean_squared_error  # Updated for modern scikit-learn compatibility
import logging
import os
import sys
from dotenv import load_dotenv

# Load local .env variables if present (useful for fallback or custom targets)
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"Data file not found at {file_path}")
        sys.exit(1)
    return pd.read_csv(file_path)

def train_and_track():
    # 1. Databricks Configuration
    # Automatically tracks using your local CLI config OR GitHub action secrets
    mlflow.set_tracking_uri("databricks")
    
    # 2. Experiment Path
    mlflow.set_experiment("/Shared/Demand_Forecasting_Baseline")

    df = load_data(os.path.join("data", "processed", "baseline_train.csv"))
    X = df[['store', 'item', 'day_of_week', 'month', 'year', 'rolling_7d_sales']]
    y = df['sales']

    with mlflow.start_run() as run:
        logger.info(f"Started MLflow run in Databricks. Run ID: {run.info.run_id}")
        
        model = xgb.XGBRegressor(n_estimators=100)
        model.fit(X, y)
        preds = model.predict(X)
        
        # Calculate RMSE using the updated scikit-learn standard
        rmse = root_mean_squared_error(y, preds)
        
        mlflow.log_metric("rmse", rmse)
        
        # Log the model artifacts directly to Databricks DBFS/Unity Catalog
        mlflow.xgboost.log_model(model, "model")
        
        logger.info(f"Run complete. RMSE: {rmse}")
        logger.info("Model logged successfully to Databricks MLflow Workspace.")

if __name__ == '__main__':
    train_and_track()