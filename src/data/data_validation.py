# Importing the libraries
import numpy as np
import pandas as pd
import yaml
import argparse
import typing
import great_expectations as ge

# create a function to read params from yaml files
def read_params(config_path: str):
    """
    Reads the parameters from the .yaml file
    input: params.yaml location
    output: paramaeters in a dictionary
    """

    with open(config_path, 'r') as yaml_file:
        try:
            config = yaml.safe_load(yaml_file)
        except yaml.YAMLError as exc:
            print(exc)
    return config

# create a function to read data from csv files
def load_data(data_path: str, data_type = 'train', model_var: typing.List[str] = None):
    """
    Reads the data from the .csv file from given path
    input: data_path, data_type
    output: data in a pandas dataframe
    """
    if data_type == 'train':
        data = ge.read_csv(data_path, 
                            sep=',', 
                            encoding='utf-8')
        data = data[model_var]
        return data


# create a function to load raw data from path
def load_raw_data(config_path: str):
    """
    Load datafrom external location (data/external) tp raw folder (data/raw)
    with train and test data
    input: data_path
    output: save file in data/raw folder
    """
    config = read_params(config_path)
    external_data_path = config['external_data_config']['external_data_csv']
    new_raw_data_path = config['raw_data_config']['new_train_data_csv']
    model_var = config['raw_data_config']['model_var']

    curr = load_data(data_path=external_data_path, 
                    data_type='train', 
                    model_var=model_var)
    
    ref = load_data(data_path=new_raw_data_path, 
                    data_type='train', 
                    model_var=model_var)
    
    return curr, ref


def save_expectations(df, config_path: str):
    """
    Saves the expectations in the .yaml file
    input: config_path
    output: save file in data/expectations folder
    """
    config = read_params(config_path)
    expectations_path = config['great_exp']['exp_json_file']
    df.save_expectation_suite(expectations_path)

def load_expectations(config_path: str):
    """
    """
    config = read_params(config_path)
    expectations_path = config['great_exp']['exp_json_file']
    return expectations_path

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', 
                        type=str, 
                        default='params.yaml')
    args = parser.parse_args()
    config_path = args.config
    curr, ref = load_raw_data(config_path)
    
    # Basic table expectation
    min_table_length = 200
    max_table_length = 3500
    curr.expect_table_row_count_to_be_between(min_table_length, max_table_length)

    curr.expect_column_values_to_be_of_type('churn', 'object')

    expected_jobs = ['no','yes']
    curr.expect_column_values_to_be_in_set('churn', expected_jobs)

    expected_churn_partition = ge.dataset.util.categorical_partition_data(curr.churn)

    curr.expect_column_chisquare_test_p_value_to_be_greater_than('churn', expected_churn_partition)
    #print(curr.get_expectation_suite())

    # Saving expectations suite for future reference
    save_expectations(curr, config_path)

    # Loading expectations suite and validate data
    exp_suite = load_expectations(config_path)
    validation_result = ref.validate(exp_suite)

    if validation_result["success"]:
        print("Validation successful!")
    else:
        raise Exception("Validation failed!")