import great_expectations as gx

context = gx.get_context()

# Create/Update Suite
suite = context.add_or_update_expectation_suite("processed_suite")

# Get asset from context
datasource = context.get_datasource("my_pandas_datasource")
asset = datasource.get_asset("baseline_train")
batch_request = asset.build_batch_request()

# Create validator
validator = context.get_validator(
    batch_request=batch_request,
    expectation_suite=suite,
)

# Add expectations
validator.expect_column_values_to_be_between(column="sales", min_value=0)
validator.expect_column_values_to_be_between(column="store", min_value=1, max_value=10)
validator.expect_column_values_to_not_be_null(column="rolling_7d_sales")

# Save
validator.save_expectation_suite(discard_failed_expectations=False)
print("Suite 'processed_suite' created successfully.")