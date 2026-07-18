# Demand Forecasting MLOps

A reference project demonstrating an end-to-end demand forecasting MLOps workflow using Python, MLflow, Databricks, Great Expectations, DVC, and FastAPI.

## Project Overview

This repository includes a complete pipeline for:
- extracting and transforming raw retail sales data
- validating processed features
- training and tracking an XGBoost demand forecasting model with MLflow
- registering the model using Databricks Unity Catalog
- serving forecasts through a FastAPI model serving endpoint
- simulating production traffic against the API
- detecting drift with Evidently and triggering a retraining workflow
- running a GitHub Actions pipeline for continuous training and deployment

## Repository Structure

- `src/etl.py` - Extract-transform-load pipeline for raw sales data into feature datasets
- `src/validate.py` - Data quality validation using Great Expectations expectations
- `src/train.py` - Model training and MLflow tracking + Unity Catalog registration logic
- `src/predict.py` - Batch inference script that loads the latest registered model and scores live data
- `src/check.py` - MLflow connectivity test script
- `src/evidently_monitor.py` - Drift monitoring script using Evidently with optional GitHub workflow dispatch
- `src/simulate_traffic.py` - Simulates API traffic against the FastAPI prediction endpoint
- `api/main.py` - FastAPI application for serving demand forecasts
- `Dockerfile` - Container image for the FastAPI prediction service
- `.github/workflows/retrain.yml` - GitHub Actions pipeline for retraining and publishing the model
- `requirements.txt` - Python dependencies for the project
- `data/` - Data storage and DVC-managed dataset pointers
- `gx/` - Great Expectations configuration, expectations, and checkpoint definitions

## Prerequisites

- Python 3.11
- pip
- Docker (for API containerization)
- DVC for dataset versioning
- Access to the configured MLflow/Databricks environment
- AWS credentials to pull DVC-managed data from the remote backend
- GitHub token and repository metadata when using drift-triggered pipeline dispatch

## Environment Variables

Create a `.env` file or set the following environment variables in your shell before running the project:

```bash
DATABRICKS_HOST=<your-databricks-host>
DATABRICKS_TOKEN=<your-databricks-token>
GITHUB_TOKEN=<your-github-token>
GITHUB_OWNER=<your-github-owner>
GITHUB_REPO=<your-github-repo>
```

## Setup

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install "setuptools<82.0.0" wheel
python -m pip install -r requirements.txt
```

Install DVC and pull data:

```bash
python -m pip install "dvc[s3]"
dvc pull
```

## Running the Pipeline

### 1. ETL

Generate processed feature datasets from raw input data:

```bash
python src/etl.py
```

This creates:
- `data/processed/baseline_train.csv`
- `data/processed/live_stream.csv`

### 2. Validate Data

Run dataset validation:

```bash
python src/validate.py
```

The repository also includes a Great Expectations checkpoint at `gx/checkpoints/primary_checkpoint.yml`.

### 3. Train and Register Model

Train the XGBoost model, log metrics/artifacts to MLflow, and register the model version in Unity Catalog:

```bash
python src/train.py
```

### 4. Batch Predictions

Score the latest live feature dataset and write results to `data/predictions/forecast_results.csv`:

```bash
python src/predict.py
```

### 5. Test MLflow Connectivity

Quickly verify MLflow tracking connectivity:

```bash
python src/check.py
```

## API Serving

### Run Locally

Start the FastAPI server from the repository root:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

### Build and Run with Docker

```bash
docker build -t demand-api:latest .
docker run --rm -p 8000:8000 demand-api:latest
```

### Prediction Endpoint

- `POST /predict` - returns forecasted demand for a single feature payload
- `GET /health` - returns health status and whether the model was loaded successfully

Example payload:

```json
{
  "store": 1,
  "item": 1,
  "day_of_week": 0,
  "month": 1,
  "year": 2017,
  "rolling_7d_sales": 123.45
}
```

## Simulate Production Traffic

Use the traffic simulator to send requests to the local API and compute average error across a sample period:

```bash
python src/simulate_traffic.py
```

Make sure the API is running at `http://127.0.0.1:8000` before using this script.

## Drift Detection and Retraining

The repository includes a drift monitor that generates a data drift report and, if drift exceeds a configured threshold, triggers a repository dispatch event to start retraining.

```bash
python src/evidently_monitor.py
```

The monitor expects:
- `data/processed/baseline_train.csv` as the reference dataset
- `data/processed/live_stream.csv` as the current dataset

## GitHub Actions Continuous Training

The workflow in `.github/workflows/retrain.yml` is configured to run on:
- `repository_dispatch` events of type `drift_detected`
- manual workflow dispatch

It performs:
- dependency installation
- DVC data pull
- data validation via Great Expectations
- model training and Unity Catalog registration
- Docker build and image push

## Notes

- `api/main.py` is configured to load a model from `http://host.docker.internal:5000` using the MLflow alias `models:/Demand_Forecasting_Model@staging`.
- `src/train.py` and `src/predict.py` are set up for a Databricks-backed MLflow environment.
- Update the environment variables and MLflow URIs to match your deployment and registry setup.

## Data Files

- `data/raw/train.csv.dvc` - raw source data pointer managed by DVC
- `data/processed/baseline_train.csv.dvc` - processed training dataset pointer
- `data/processed/live_stream.csv.dvc` - processed live scoring dataset pointer
- `data/predictions/forecast_results.csv` - example forecast output file

## Contributing

Contributions are welcome. Please open issues or pull requests for bug fixes, improvements, or new features.
