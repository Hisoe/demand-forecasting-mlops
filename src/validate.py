# src/validate.py
import great_expectations as ge
import sys
import os
import logging

# Engineering Standard: Configure logging instead of using print()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def validate_dataset(file_path: str) -> bool:
    """Runs quality expectations against a dataset to ensure schema integrity."""
    logger.info(f"Loading data for validation: {file_path}")
    
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False

    # Load data directly into the Great Expectations wrapper
    df = ge.read_csv(file_path)
    
    logger.info("Executing expectation suite...")
    
    # 1. Domain Constraint: Sales cannot be negative
    res_sales = df.expect_column_values_to_be_between(column="sales", min_value=0)
    
    # 2. Schema Constraint: Store IDs must be exactly 1 through 10
    res_store = df.expect_column_values_to_be_between(column="store", min_value=1, max_value=10)
    
    # 3. Completeness Constraint: Our engineered feature must not contain nulls
    res_rolling = df.expect_column_values_to_not_be_null(column="rolling_7d_sales")
    
    # Aggregate boolean success flags
    is_successful = res_sales["success"] and res_store["success"] and res_rolling["success"]
    return is_successful

def run_validation():
    """Orchestrates the validation step for the pipeline."""
    baseline_path = os.path.join("data", "processed", "baseline_train.csv")
    
    if validate_dataset(baseline_path):
        logger.info("Validation Passed: Dataset meets production quality standards.")
        sys.exit(0)  # Exit code 0 tells CI/CD the pipeline can proceed
    else:
        logger.error("Validation Failed: Data contract violated. Halting pipeline.")
        sys.exit(1)  # Exit code 1 forcefully stops the CI/CD pipeline

if __name__ == "__main__":
    run_validation()