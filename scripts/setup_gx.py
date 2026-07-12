import great_expectations as gx

context = gx.get_context()

# 1. Add the datasource
datasource = context.sources.add_or_update_pandas(name="my_pandas_datasource")

# 2. Add your assets
datasource.add_csv_asset(name="raw_train", filepath_or_buffer="data/raw/train.csv")
datasource.add_csv_asset(name="baseline_train", filepath_or_buffer="data/processed/baseline_train.csv")
datasource.add_csv_asset(name="live_stream", filepath_or_buffer="data/processed/live_stream.csv")

# 3. CRITICAL: Persist the configuration to disk
# This ensures that your great_expectations.yml file is actually updated
context.add_or_update_datasource(datasource=datasource)

print("Datasource and Assets configured and saved successfully.")