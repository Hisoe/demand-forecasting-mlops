import pandas as pd
import json
import logging
import requests
import os
from dotenv import load_dotenv
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def trigger_retrain_pipeline():
    token = os.getenv("GITHUB_TOKEN")
    owner = os.getenv("GITHUB_OWNER")
    repo = os.getenv("GITHUB_REPO")
    
    url = f"https://api.github.com/repos/{owner}/{repo}/dispatches"
    headers = {
        "Authorization": f"token {token}",
        "Accept": "application/vnd.github.v3+json"
    }
    data = {"event_type": "drift_detected"}
    
    response = requests.post(url, json=data, headers=headers)
    if response.status_code == 204:
        logger.info("Successfully triggered CI/CD pipeline!")
    else:
        logger.error(f"Trigger failed: {response.status_code} - {response.text}")

def generate_drift_report():
    # 1. Load Data
    ref = pd.read_csv("data/processed/baseline_train.csv")
    curr = pd.read_csv("data/processed/live_stream.csv")

    # 2. Setup Report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=curr)
    
    # 3. Analyze Results
    report_dict = report.as_dict()
    
    # Calculate percentage of drifting features
    metrics = report_dict['metrics'][0]['result']
    num_features = len(metrics['drift_by_columns'])
    drifted_features = [col for col, data in metrics['drift_by_columns'].items() if data['drift_detected']]
    drift_share = len(drifted_features) / num_features
    
    logger.info(f"Drift share: {drift_share:.2%}")

    # 4. THRESHOLD LOGIC: Trigger only if > 30% of features drift
    # This prevents 'thrashing' from minor noise
    if drift_share > 0.30:
        logger.warning(f"Significant drift detected in {len(drifted_features)} features.")
        report.save_html("model_drift_dashboard.html")
        trigger_retrain_pipeline()
    else:
        logger.info("Drift is within acceptable limits. No action taken.")

if __name__ == "__main__":
    generate_drift_report()