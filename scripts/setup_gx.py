import great_expectations as gx
from pathlib import Path

context = gx.get_context()

# 1. Define paths using Path objects for cross-platform compatibility
# The project root is the current working directory
project_root = Path.cwd()

# Define paths explicitly using / as the join operator (Path handles the OS translation)
raw_train_path = project_root / "data" / "raw" / "train.csv"
baseline_train_path = project_root / "data" / "processed" / "baseline_train.csv"
live_stream_path = project_root / "data" / "processed" / "live_stream.csv"

# 2. Add the datasource
datasource = context.sources.add_or_update_pandas(name="my_pandas_datasource")

# 3. Add assets (Convert to string, but Path ensures correct platform formatting)
datasource.add_csv_asset(name="raw_train", filepath_or_buffer=str(raw_train_path))
datasource.add_csv_asset(name="baseline_train", filepath_or_buffer=str(baseline_train_path))
datasource.add_csv_asset(name="live_stream", filepath_or_buffer=str(live_stream_path))

# 4. Save
context.add_or_update_datasource(datasource=datasource)

print(f"Datasource configured with paths relative to: {project_root}")