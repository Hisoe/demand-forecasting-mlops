import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import root_mean_squared_error
import logging
import os
import sys
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
    
    os.environ["MLFLOW_USE_DATABRICKS_SDK_MODEL_ARTIFACTS_REPO_FOR_UC"] = "true"
    
    # 1. Connect to your remote Databricks Workspace
    mlflow.set_tracking_uri("databricks")
    
    # 2. Tell MLflow to route models to the modern Unity Catalog instead of workspace storage
    mlflow.set_registry_uri("databricks-uc")
    
    # 3. Define where the experiment visual metrics are logged
    mlflow.set_experiment("/Shared/Demand_Forecasting_Baseline")

    # 4. Strict Unity Catalog 3-part namespace syntax
    # Format: catalog.schema.model_name
    registered_model_name = "workspace.default.demand_forecasting_baseline"

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
        
        # Calculate metric with modern scikit-learn compatibility
        rmse = root_mean_squared_error(y, preds)
        mlflow.log_metric("rmse", rmse)
        
        # 5. Log the model files AND dynamically register it under the UC schema
        logger.info(f"Logging and registering model to Unity Catalog as: {registered_model_name}")
        
        # Grab a small sample (e.g., first 5 rows) of your training features for the signature
        input_example = X.head(5)
        
        mlflow.xgboost.log_model(
            xgb_model=model,
            artifact_path="model",
            registered_model_name=registered_model_name,
            input_example=input_example  
        )

if __name__ == '__main__':
    train_and_track()