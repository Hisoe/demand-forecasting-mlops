import os
# Force the SDK to use the proxy
os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "True"

import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import root_mean_squared_error
import logging
import sys
import shutil
from dotenv import load_dotenv

# Import the Databricks SDK for secure API uploads
from databricks.sdk import WorkspaceClient

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

def upload_directory_to_volume(local_dir, remote_volume_dir):
    """
    Safely uploads a local directory of files to a Databricks Volume 
    using the official Databricks API Client.
    """
    logger.info(f"Uploading local artifacts from {local_dir} to Volume path {remote_volume_dir} via Databricks API...")
    db_client = WorkspaceClient()
    
    for root, _, files in os.walk(local_dir):
        for file in files:
            local_file_path = os.path.join(root, file)
            # Create relative structure for remote storage matching the local subdirectory
            relative_path = os.path.relpath(local_file_path, local_dir)
            remote_file_path = os.path.join(remote_volume_dir, relative_path).replace("\\", "/")
            
            logger.info(f"Uploading {relative_path} -> {remote_file_path}")
            # Upload the individual file bytes over HTTPS
            db_client.files.upload_from(
                file_path=remote_file_path,
                source_path=local_file_path,
                overwrite=True
            )

def train_and_track():
    # 1. Connect to your remote Databricks Workspace
    mlflow.set_tracking_uri("databricks")
    
    # 2. Tell MLflow to route models to Unity Catalog
    mlflow.set_registry_uri("databricks-uc")
    
    # 3. Define where metrics are logged
    mlflow.set_experiment("/Shared/Demand_Forecasting_Baseline")

    # 4. Strict Unity Catalog 3-part namespace syntax
    registered_model_name = "workspace.default.demand_forecasting_baseline"

    # Volume path (Must be prefixed with /Volumes for API schema compatibility)
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
        
        logger.info("Saving model artifacts locally...")
        
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

        # 6. Upload the files safely using the Databricks API
        # This completely avoids local mount directory PermissionErrors!
        upload_directory_to_volume(local_temp_dir, volume_model_path)
        
        # 7. Register the model using the Volume URI
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