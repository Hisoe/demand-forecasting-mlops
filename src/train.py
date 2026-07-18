import os
import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import root_mean_squared_error
import logging
import sys
import requests
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    if not os.path.exists(file_path):
        logger.error(f"Data file not found at {file_path}")
        sys.exit(1)
    return pd.read_csv(file_path)

def train_and_track():
    # 1. Initialize MLflow tracking to the workspace
    mlflow.set_tracking_uri("databricks")
    mlflow.set_experiment("/Shared/Demand_Forecasting_Baseline")

    registered_model_name = "workspace.default.demand_forecasting_baseline"

    df = load_data(os.path.join("data", "processed", "baseline_train.csv"))
    X = df[['store', 'item', 'day_of_week', 'month', 'year', 'rolling_7d_sales']]
    y = df['sales']

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        logger.info(f"Started CLEAN MLflow run. Run ID: {run_id}")
        
        # Train model
        model = xgb.XGBRegressor(n_estimators=100)
        model.fit(X, y)
        preds = model.predict(X)
        
        rmse = root_mean_squared_error(y, preds)
        mlflow.log_metric("rmse", rmse)
        
        # 2. Log model artifacts to the active run ONLY
        # Notice: registered_model_name is omitted here to prevent client-side S3 calls
        logger.info("LOGGING ARTIFACTS TO RUN ONLY - S3 REGISTER BYPASS")
        input_example = X.head(5)
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            input_example=input_example
        )

        # 3. Handle model registration server-side via direct Unity Catalog REST API
        logger.info(f"Requesting Databricks to register model server-side in Unity Catalog: {registered_model_name}")
        
        host = os.getenv("DATABRICKS_HOST").rstrip("/")
        token = os.getenv("DATABRICKS_TOKEN")
        
        # Target the explicit Unity Catalog model versions endpoint
        endpoint = f"{host}/api/2.1/unity-catalog/models/versions"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
        
        # The exact payload schema accepted by the Unity Catalog 2.1 API
        payload = {
            "model_name": registered_model_name,  # <-- The missing required key (workspace.default.demand_forecasting_baseline)
            "source": f"runs:/{run_id}/model",
            "run_id": run_id
        }
        
        response = requests.post(endpoint, headers=headers, json=payload)
        
        if response.status_code == 200 or response.status_code == 201:
            logger.info(f"Success! Registered model version in Unity Catalog: {response.json()}")
        else:
            logger.error(f"Server registration failed: Status {response.status_code}, Error: {response.text}")
            sys.exit(1)

if __name__ == '__main__':
    train_and_track()