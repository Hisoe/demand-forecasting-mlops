import great_expectations as gx

context = gx.get_context()
# This will show you if the context knows about your datasource
print(f"Datasources: {context.fluent_datasources}")