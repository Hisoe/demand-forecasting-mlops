# src/simulate_traffic.py
import pandas as pd
import requests
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# The URL of your locally running FastAPI server
API_URL = "http://127.0.0.1:8000/predict"

def run_simulation():
    logger.info("Starting production traffic simulation...")
    
    # Load the unseen 2017 data
    df = pd.read_csv("data/processed/live_stream.csv")
    
    # We will simulate just the first 50 days of 2017 for Store 1, Item 1 to see how it performs
    simulation_data = df[(df['store'] == 1) & (df['item'] == 1)].head(50)
    
    errors = []

    for index, row in simulation_data.iterrows():
        # 1. Prepare the JSON payload
        payload = {
            "store": int(row['store']),
            "item": int(row['item']),
            "day_of_week": int(row['day_of_week']),
            "month": int(row['month']),
            "year": int(row['year']),
            "rolling_7d_sales": float(row['rolling_7d_sales'])
        }
        
        actual_sales = row['sales']
        
        try:
            # 2. Send the request to the FastAPI server
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                prediction = response.json()['forecasted_sales']
                
                # 3. Calculate the error (Drift indicator)
                error = abs(prediction - actual_sales)
                errors.append(error)
                
                logger.info(f"Date: 2017-{int(row['month']):02d}-{(index % 31) + 1:02d} | Predicted: {prediction:.1f} | Actual: {actual_sales} | Error: {error:.1f}")
            else:
                logger.error(f"API Error: {response.text}")
                
        except requests.exceptions.ConnectionError:
            logger.error("Failed to connect to the API. Is your FastAPI server running?")
            break
            
        # Pause briefly to simulate real-time API traffic
        time.sleep(0.5)
        
    if errors:
        avg_error = sum(errors) / len(errors)
        logger.info(f"--- Simulation Complete ---")
        logger.info(f"Average Error (MAE) over 50 days: {avg_error:.2f}")
        logger.info(f"Remember: Our acceptable training RMSE threshold was 25.0.")

if __name__ == "__main__":
    run_simulation()