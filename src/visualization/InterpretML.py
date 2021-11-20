import numpy as np
import pandas as pd
import yaml
import argparse
import typing
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score,recall_score,accuracy_score,precision_score,confusion_matrix,classification_report
from interpret.glassbox import ExplainableBoostingClassifier
from interpret import show, preserve, show_link, set_show_addr
from interpret.provider import InlineProvider
from interpret import set_visualize_provider

set_visualize_provider(InlineProvider())

def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def accuracymeasures(y_test,predictions,avg_method):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average=avg_method)
    recall = recall_score(y_test, predictions, average=avg_method)
    f1score = f1_score(y_test, predictions, average=avg_method)
    target_names = ['0','1']
    print("Classification report")
    print("---------------------","\n")
    print(classification_report(y_test, predictions,target_names=target_names),"\n")
    print("Confusion Matrix")
    print("---------------------","\n")
    print(confusion_matrix(y_test, predictions),"\n")

    print("Accuracy Measures")
    print("---------------------","\n")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1score)

    return accuracy,precision,recall,f1score

def get_feat_and_target(df,target):
    """
    Get features and target variables seperately from given dataframe and target 
    input: dataframe and target column
    output: two dataframes for x and y 
    """
    x=df.drop(target,axis=1)
    y=df[[target]]
    return x,y    

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["target"]
    max_depth=config["random_forest"]["max_depth"]
    n_estimators=config["random_forest"]["n_estimators"]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    train_x,train_y=get_feat_and_target(train,target)
    test_x,test_y=get_feat_and_target(test,target)

    return train_x,train_y,test_x,test_y

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_x,train_y,test_x,test_y = train_and_evaluate(config_path=parsed_args.config)

    ebm = ExplainableBoostingClassifier()
    ebm.fit(train_x, train_y)

    ebm_global = ebm.explain_global(name = 'EBM')
    #preserve(ebm_global, 'number_customer_service_calls', 'number_customer_service_calls.html')
    
    """ 
    'show' function does not work for .py files. We need to run the code in jupyter notebook.
    """
    #set_show_addr(("127.0.0.1", 5000))
    show(ebm_global)
    plt.show()


