# src/etl.py
import pandas as pd
import os

def extract(file_path: str) -> pd.DataFrame:
    """Extracts data from the source."""
    print(f"Extracting raw data from {file_path}...")
    df = pd.read_csv(file_path)
    return df

def transform(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Transforms raw data into engineered features and splits chronologically."""
    print("Engineering time-series features...")
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    
    # Create a 7-day rolling average
    df['rolling_7d_sales'] = df.groupby(['store', 'item'])['sales'].transform(lambda x: x.rolling(7).mean())
    df.dropna(inplace=True)
    
    print("Splitting data into Baseline (2013-2016) and Live Stream (2017)...")
    baseline_df = df[df['year'] < 2017].copy()
    live_stream_df = df[df['year'] == 2017].copy()
    
    return baseline_df, live_stream_df

def load(baseline_df: pd.DataFrame, live_stream_df: pd.DataFrame, output_dir: str):
    """Loads the transformed data into the target destination."""
    print(f"Loading processed files to {output_dir}...")
    os.makedirs(output_dir, exist_ok=True)
    
    baseline_path = os.path.join(output_dir, "baseline_train.csv")
    live_stream_path = os.path.join(output_dir, "live_stream.csv")
    
    baseline_df.to_csv(baseline_path, index=False)
    live_stream_df.to_csv(live_stream_path, index=False)
    
def run_pipeline():
    """Orchestrates the ETL process."""
    raw_path = os.path.join("data", "raw", "train.csv")
    output_directory = os.path.join("data", "processed")
    
    # The ETL execution flow
    raw_df = extract(raw_path)
    baseline_df, live_stream_df = transform(raw_df)
    load(baseline_df, live_stream_df, output_directory)
    
    print("Success! ETL pipeline completed.")

if __name__ == "__main__":
    run_pipeline()