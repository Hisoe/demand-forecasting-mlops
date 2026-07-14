import great_expectations as gx

context = gx.get_context()

# 1. Define paths as RELATIVE strings
# Great Expectations will resolve these relative to the project root
raw_train_path = "data/raw/train.csv"
baseline_train_path = "data/processed/baseline_train.csv"
live_stream_path = "data/processed/live_stream.csv"

# 2. Add the datasource
datasource = context.sources.add_or_update_pandas(name="my_pandas_datasource")

# 3. Add assets using the relative strings
datasource.add_csv_asset(name="raw_train", filepath_or_buffer=raw_train_path)
datasource.add_csv_asset(name="baseline_train", filepath_or_buffer=baseline_train_path)
datasource.add_csv_asset(name="live_stream", filepath_or_buffer=live_stream_path)

# 4. Persist
context.add_or_update_datasource(datasource=datasource)

print("Datasource configured with relative paths.")