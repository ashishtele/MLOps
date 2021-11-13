# Import libraries
import os
import argparse
import pandas as pd
import numpy as np
from load_data import read_params
from sklearn.model_selection import train_test_split
import typing

# create a function to split the data
def split_data(data: pd.DataFrame, 
                train_data_path: str,
                test_data_path: str,
                split_ratio: float,
                random_state: int):
    # Split the data into training and test sets
    train, test = train_test_split(data,
                                    test_size=split_ratio,
                                    random_state=random_state)
    train.to_csv(train_data_path, 
                sep=",",
                index=False,
                encoding='utf-8')
    test.to_csv(train_data_path, 
                sep=",",
                index=False,
                encoding='utf-8')


def split_and_saved_data(config_path: str):
    """
    Split the train dataset (data/raw) and save it in the data/processed folder 
    input: config path
    output: save splitted files in output folder
    """
    # Read the config file
    config = read_params(config_path)
    # Read the data
    raw_data_path = config['raw_data_config']['raw_data_csv']
    test_data_path = config['processed_data_config']['test_data_csv']
    train_data_path = config['processed_data_config']['train_data_csv']
    split_ratio = config['raw_data_config']['train_test_split_ratio']
    random_state = config['raw_data_config']['random_state']
    # Read the data
    raw_df = pd.read_csv(raw_data_path)
    # Split the data
    split_data(raw_df,
                 train_data_path, 
                 test_data_path, 
                 split_ratio, 
                 random_state)

if __name__ == 'main':
    args=argparse.ArgumentParser()
    args.add_argument("--config", 
                        type=str, 
                        default="params.yaml")
    parsed_args = args.parse_args()
    split_and_saved_data(parsed_args.config)