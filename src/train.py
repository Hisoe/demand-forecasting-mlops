import os
# Force the SDK to use the proxy just in case
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "True"

import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import root_mean_squared_error
import logging
import sys
import shutil  # Clean directory utilities
from dotenv import load_dotenv

# Load local environment variables if available
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
    # 1. Connect to your remote Databricks Workspace
    mlflow.set_tracking_uri("databricks")
    
    # 2. Tell MLflow to route models to Unity Catalog
    mlflow.set_registry_uri("databricks-uc")
    
    # 3. Define where metrics are logged
    mlflow.set_experiment("/Shared/Demand_Forecasting_Baseline")

    # 4. Strict Unity Catalog 3-part namespace syntax
    registered_model_name = "workspace.default.demand_forecasting_baseline"

    # Define your volume folder path
    # Databricks automatically exposes managed volumes to external API uploads securely!
    volume_model_path = "/Volumes/workspace/default/demand_forecasting_volumes/demand_model"

    # Load data features
    df = load_data(os.path.join("data", "processed", "baseline_train.csv"))
    X = df[['store', 'item', 'day_of_week', 'month', 'year', 'rolling_7d_sales']]
    y = df['sales']

    with mlflow.start_run() as run:
        logger.info(f"Started MLflow run in Databricks. Run ID: {run.info.run_id}")
        
        # Initialize and fit XGBoost Regressor
        model = xgb.XGBRegressor(n_estimators=100)
        model.fit(X, y)
        preds = model.predict(X)
        
        # Calculate metric
        rmse = root_mean_squared_error(y, preds)
        mlflow.log_metric("rmse", rmse)
        
        logger.info("Registering model through safe Unity Catalog Volume...")
        
        # Clean local temporary model directory if it exists
        local_temp_dir = "temp_model_artifacts"
        if os.path.exists(local_temp_dir):
            shutil.rmtree(local_temp_dir)

        # Grab a small sample for the schema signature
        input_example = X.head(5)

        # 5. Save the MLmodel structure locally
        mlflow.xgboost.save_model(
            xgb_model=model,
            path=local_temp_dir,
            input_example=input_example
        )

        # 6. Safely transfer the local model directory into the Managed Volume
        # This completely avoids AWS S3 PutObject calls and uploads directly over the Databricks API!
        if os.path.exists(volume_model_path):
            shutil.rmtree(volume_model_path)
        shutil.copytree(local_temp_dir, volume_model_path)
        
        # 7. Register the model using the Volume URI
        # MLflow will link Unity Catalog to the volume file storage structure natively
        mlflow.register_model(
            model_uri=f"dbfs:{volume_model_path}",
            name=registered_model_name
        )
        
        # Clean up local workspace artifacts
        if os.path.exists(local_temp_dir):
            shutil.rmtree(local_temp_dir)
            
        logger.info("Successfully registered model under workspace.default.demand_forecasting_baseline!")

if __name__ == '__main__':
    train_and_track()