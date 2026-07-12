import great_expectations as gx
import os

# Get the directory of the current script (scripts/)
script_dir = os.path.dirname(os.path.abspath(__file__))
# Navigate one level up to the root (D:\demand-forecasting-mlops\)
root_dir = os.path.dirname(script_dir)

context = gx.get_context()

# 1. Add the datasource
datasource = context.sources.add_or_update_pandas(name="my_pandas_datasource")

# 2. Add your assets using absolute paths derived from the root_dir
datasource.add_csv_asset(
    name="raw_train", 
    filepath_or_buffer=os.path.join(root_dir, "data", "raw", "train.csv")
)
datasource.add_csv_asset(
    name="baseline_train", 
    filepath_or_buffer=os.path.join(root_dir, "data", "processed", "baseline_train.csv")
)
datasource.add_csv_asset(
    name="live_stream", 
    filepath_or_buffer=os.path.join(root_dir, "data", "processed", "live_stream.csv")
)

print("Datasource and Assets configured successfully using absolute paths.")