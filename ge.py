import great_expectations as ge

context = ge.data_context.DataContext()

suite = context.create_expectation_suite(
    expectation_suite_name="check_raw_data",
    overwrite_existing = True
)

batch_kwargs = {
    'path': 'data/raw/train.csv',
    'datasource': 'data_dir',
    'data_asset_name': 'train',
    'reader_method': 'read_csv'
}
batch = context.get_batch(batch_kwargs, suite)
#print(batch.head())

# Expect column to exists
batch.expect_column_to_exist('churn')
batch.expect_column_values_to_not_be_null('churn')
batch.expect_column_values_to_be_of_type('total_day_calls', 'int')
batch.expect_column_distinct_values_to_be_in_set('churn', ['no', 'yes'])

# print expectation_suite
#print(batch.get_expectation_suite())

batch.save_expectation_suite()
results = context.run_validation_operator('action_list_operator', assets_to_validate=[batch])

#context.open_data_docs()

context.run_checkpoint(checkpoint_name="my_base_check")
