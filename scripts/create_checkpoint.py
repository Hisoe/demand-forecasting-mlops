import great_expectations as gx

# 1. Get the context
context = gx.get_context()

# 2. Retrieve the existing datasource and asset
# No path changes needed; this pulls the config from the GX context
datasource = context.get_datasource("my_pandas_datasource")
asset = datasource.get_asset("baseline_train")
batch_request = asset.build_batch_request()

# 3. Create or Update the Checkpoint
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

# 4. Run it to verify everything is working
result = checkpoint.run()
if result.success:
    print("✅ Validation successful!")
else:
    print("❌ Validation failed!")