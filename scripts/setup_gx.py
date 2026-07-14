import great_expectations as gx
from pathlib import Path
import os

# 1. Setup GX
context = gx.get_context()
datasource = context.sources.add_or_update_pandas(name="my_pandas_datasource")

assets = {
    "raw_train": "data/raw/train.csv",
    "baseline_train": "data/processed/baseline_train.csv",
    "live_stream": "data/processed/live_stream.csv"
}

for name, path in assets.items():
    datasource.add_csv_asset(name=name, filepath_or_buffer=path)

# 2. Persist the configuration (This generates the backslashes)
context.add_or_update_datasource(datasource=datasource)

# 3. SURGICAL OVERRIDE: Open the file as plain text and force forward slashes
# This bypasses the YAML library entirely.
config_path = Path("gx/great_expectations.yml")

if config_path.exists():
    content = config_path.read_text(encoding='utf-8')
    
    # Force replace ALL backslashes that appear in path-like strings
    # We replace \\ with / (because YAML escapes backslashes)
    fixed_content = content.replace("\\\\", "/").replace("\\", "/")
    
    config_path.write_text(fixed_content, encoding='utf-8')
    print("Successfully forced forward slashes into gx/great_expectations.yml")