import great_expectations as gx
import os
from pathlib import Path

# 1. Setup the context
context = gx.get_context()

# 2. Use dummy strings that don't look like file paths to the library
# This prevents the library from trying to "helpfully" normalize them.
datasource = context.sources.add_or_update_pandas(name="my_pandas_datasource")
datasource.add_csv_asset(name="raw_train", filepath_or_buffer="PLACEHOLDER_RAW")
datasource.add_csv_asset(name="baseline_train", filepath_or_buffer="PLACEHOLDER_BASELINE")
datasource.add_csv_asset(name="live_stream", filepath_or_buffer="PLACEHOLDER_LIVE")

context.add_or_update_datasource(datasource=datasource)

# 3. NOW, use plain-text replacement to force the slashes
config_path = Path("gx/great_expectations.yml")
content = config_path.read_text(encoding='utf-8')

# Replace the placeholders with the actual, clean paths
content = content.replace("PLACEHOLDER_RAW", "data/raw/train.csv")
content = content.replace("PLACEHOLDER_BASELINE", "data/processed/baseline_train.csv")
content = content.replace("PLACEHOLDER_LIVE", "data/processed/live_stream.csv")

# 4. Write back as plain text
config_path.write_text(content, encoding='utf-8')

print("Configuration updated using placeholder injection.")