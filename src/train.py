import pandas as pd
import xgboost as xgb
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_absolute_error, mean_squared_error
import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_data(file_path):
    logger.info(f"Loading data from {file_path}")
    if not os.path.exists(file_path):
        sys.exit(1)
    return pd.read_csv(file_path)

def train_and_track():
    tracking_uri = os.getenv('MLFLOW_TRACKING_URI', 'file:./mlruns')
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("Demand_Forecasting_Baseline")

    df = load_data(os.path.join("data", "processed", "baseline_train.csv"))
    X = df[['store', 'item', 'day_of_week', 'month', 'year', 'rolling_7d_sales']]
    y = df['sales']

    with mlflow.start_run():
        model = xgb.XGBRegressor(n_estimators=100)
        model.fit(X, y)
        preds = model.predict(X)
        rmse = mean_squared_error(y, preds, squared=False)
        mlflow.log_metric("rmse", rmse)
        
        mlflow.xgboost.log_model(model, "model")
        logger.info(f"Run complete. RMSE: {rmse}")

if __name__ == '__main__':
    train_and_track()