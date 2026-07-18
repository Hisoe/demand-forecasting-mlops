import os
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "True"

import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import root_mean_squared_error
import logging
import sys
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
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment("/Shared/Demand_Forecasting_Baseline")

    registered_model_name = "workspace.default.demand_forecasting_baseline"

    df = load_data(os.path.join("data", "processed", "baseline_train.csv"))
    X = df[['store', 'item', 'day_of_week', 'month', 'year', 'rolling_7d_sales']]
    y = df['sales']

    with mlflow.start_run() as run:
        logger.info(f"Started MLflow run. Run ID: {run.info.run_id}")
        
        model = xgb.XGBRegressor(n_estimators=100)
        model.fit(X, y)
        preds = model.predict(X)
        
        rmse = root_mean_squared_error(y, preds)
        mlflow.log_metric("rmse", rmse)
        
        logger.info(f"Logging and registering model to UC: {registered_model_name}")
        input_example = X.head(5)
        
        # This will now successfully proxy the files via HTTPS directly over your mock profile!
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
            input_example=input_example
        )
        logger.info("Successfully registered model under Unity Catalog!")

if __name__ == '__main__':
    train_and_track()