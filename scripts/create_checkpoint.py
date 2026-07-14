import great_expectations as gx

context = gx.get_context()

# Retrieve asset
datasource = context.get_datasource("my_pandas_datasource")
asset = datasource.get_asset("baseline_train")
batch_request = asset.build_batch_request()

# Create/Update Checkpoint
checkpoint = context.add_or_update_checkpoint(
    name="primary_checkpoint",
    validations=[
        {
            "batch_request": batch_request,
            "expectation_suite_name": "processed_suite",
        },
    ],
)

print("Checkpoint 'primary_checkpoint' created/updated successfully!")

# Verification Run
result = checkpoint.run()
if result.success:
    print("✅ Validation successful!")
else:
    print("❌ Validation failed!")