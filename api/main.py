# api/main.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow
import pandas as pd
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="Demand Forecasting API", description="MLOps Model Serving")

# Define the exact schema the API expects from clients
class ForecastRequest(BaseModel):
    store: int
    item: int
    day_of_week: int
    month: int
    year: int
    rolling_7d_sales: float

# Global variable to hold the model in memory
model = None

@app.on_event("startup")
def load_model():
    """Loads the model from MLflow Registry on server startup."""
    global model
    logger.info("Initializing API and connecting to MLflow...")
    
    # Point to your local MLflow tracking server
    mlflow.set_tracking_uri("http://localhost:5000")
    
    # The Modern Way: Fetching the model via its assigned Alias
    model_uri = "models:/Demand_Forecasting_Model@staging"
    
    try:
        logger.info(f"Downloading model from: {model_uri}")
        model = mlflow.xgboost.load_model(model_uri)
        logger.info("Model loaded successfully into API memory.")
    except Exception as e:
        logger.error(f"Failed to load model from MLflow: {e}")

@app.post("/predict")
def predict(request: ForecastRequest):
    """Generates a sales forecast based on incoming JSON data."""
    if model is None:
        raise HTTPException(status_code=503, detail="Model is currently unavailable.")
    
    try:
        # Convert the incoming JSON payload into a Pandas DataFrame
        input_data = pd.DataFrame([request.model_dump()]) 
        
        # Make the prediction
        prediction = model.predict(input_data)
        
        # Return the prediction
        return {
            "store": request.store,
            "item": request.item,
            "forecasted_sales": float(prediction[0])
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail="Error generating prediction.")

@app.get("/health")
def health_check():
    """Simple endpoint to verify the API is running."""
    return {"status": "healthy", "model_loaded": model is not None}