# src/evidently_monitor.py
import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_drift_report():
    logger.info("Loading reference and current datasets...")
    
    # 1. Load the historical training data (Reference)
    try:
        reference_data = pd.read_csv("data/processed/baseline_train.csv") 
    except FileNotFoundError:
        logger.error("Could not find train.csv. Please ensure the path to your training data is correct.")
        return

    # 2. Load the live production data (Current)
    current_data = pd.read_csv("data/processed/live_stream.csv")
    
    # To keep the dashboard fast and readable, we will analyze just Store 1, Item 1
    ref_sample = reference_data[(reference_data['store'] == 1) & (reference_data['item'] == 1)]
    curr_sample = current_data[(current_data['store'] == 1) & (current_data['item'] == 1)]
    
    logger.info("Generating Evidently AI Drift and Quality Report...")
    
    # 3. Initialize the Evidently Report with specific presets
    drift_report = Report(metrics=[
        DataQualityPreset(),
        DataDriftPreset(),
    ])
    
    # 4. Run the statistical calculations
    drift_report.run(reference_data=ref_sample, current_data=curr_sample)
    
    # 5. Save the output as a beautiful interactive HTML dashboard
    output_file = "model_drift_dashboard.html"
    drift_report.save_html(output_file)
    
    logger.info(f"Success! Dashboard generated and saved as: {output_file}")

if __name__ == "__main__":
    generate_drift_report()