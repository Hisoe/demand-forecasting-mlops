import great_expectations as gx

# 1. Get the context (Great Expectations looks for the 'gx' folder relative to where you run the script)
context = gx.get_context()

# 2. Add or update the expectation suite
suite = context.add_or_update_expectation_suite("processed_suite")

# 3. Get the datasource and asset
# These are retrieved from the context configuration, so no path changes needed here!
datasource = context.get_datasource("my_pandas_datasource")
asset = datasource.get_asset("baseline_train")
batch_request = asset.build_batch_request()

# 4. Create the validator
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite=suite,
)

# 5. Add your expectations
validator.expect_column_values_to_be_between(column="sales", min_value=0)
validator.expect_column_values_to_be_between(column="store", min_value=1, max_value=10)
validator.expect_column_values_to_not_be_null(column="rolling_7d_sales")

# 6. Save the suite
validator.save_expectation_suite(discard_failed_expectations=False)

print("Suite 'processed_suite' created and saved successfully in gx/expectations/")