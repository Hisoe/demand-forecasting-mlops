import os
import pandas as pd
import mlflow
import mlflow.xgboost
import logging
import sys
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_new_data(file_path):
    """Loads the latest batch of feature data that needs forecasting."""
    if not os.path.exists(file_path):
        logger.error(f"Inference data file not found at {file_path}")
        sys.exit(1)
    return pd.read_csv(file_path)

def run_batch_predictions():
    # 1. Configure MLflow to read from Unity Catalog
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")

    model_name = "workspace.default.demand_forecasting_baseline"
    client = mlflow.tracking.MlflowClient()
    
    logger.info(f"Querying Unity Catalog for the latest version details of {model_name}...")
    try:
        # Fetch all registered versions for this model namespace
        versions = client.search_model_versions(f"name='{model_name}'")
        if not versions:
            logger.error(f"No versions found for model {model_name} in Unity Catalog.")
            sys.exit(1)
            
        # Extract the version object with the highest version index number
        latest_version_obj = max(versions, key=lambda v: int(v.version))
        logger.info(f"Identified version index: {latest_version_obj.version}")
        
        # Bypass string URI parsing bugs by passing the direct underlying run artifact source storage path
        # Format: dbfs:/user/hive/warehouse/ws.db/... or UC storage root paths
        model_uri = latest_version_obj.source
        logger.info(f"Resolving artifact backing source: {model_uri}")
        
    except Exception as e:
        logger.error(f"Failed to resolve version source metadata from registry. Error: {str(e)}")
        sys.exit(1)
    
    logger.info(f"Loading model structure directly from backing location...")
    try:
        model = mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        logger.error(f"Failed to load model layers from Unity Catalog storage. Error: {str(e)}")
        sys.exit(1)

    # 2. Load incoming batch data
    # Swapped from baseline_inference.csv to your active live_stream.csv file
    inference_data_path = os.path.join("data", "processed", "live_stream.csv") 
    logger.info(f"Loading features for scoring from {inference_data_path}...")
    df_features = load_new_data(inference_data_path)

    feature_columns = ['store', 'item', 'day_of_week', 'month', 'year', 'rolling_7d_sales']
    X_inference = df_features[feature_columns]

    # 3. Generate forecasts
    logger.info("Generating demand forecasts...")
    predictions = model.predict(X_inference)

    # 4. Consolidate results
    df_results = df_features.copy()
    df_results['predicted_sales'] = predictions
    df_results['forecast_generated_at'] = pd.Timestamp.now()

    # 5. Output forecasts
    output_path = os.path.join("data", "predictions", "forecast_results.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_results.to_csv(output_path, index=False)
    
    logger.info(f"Batch prediction successfully complete! Results written to {output_path}")
    
if __name__ == '__main__':
    run_batch_predictions()