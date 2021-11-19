"""
This is script is to try W&B in MLOps pipeline.
"""
import json
import yaml
import joblib
import argparse
import wandb
import numpy as np
import pandas as pd
from urllib.parse import urlparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score,recall_score,accuracy_score,precision_score,confusion_matrix,classification_report

import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

wandb.login()

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

    return train_x,train_y,test_x,test_y,max_depth,n_estimators

if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()

    train_x,train_y,test_x,test_y,max_depth,n_estimators=train_and_evaluate(config_path=parsed_args.config)

    # Train the Ridge model
    model = RandomForestClassifier(max_depth=max_depth,
                                    n_estimators=n_estimators)
    model.fit(train_x, train_y)
    y_pred = model.predict(test_x)
    y_probas = model.predict_proba(test_x)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    accuracy,precision,recall,f1score = accuracymeasures(test_y,y_pred,'weighted')

    # Initialize W&B run

    run = wandb.init(project="churn_model", name="rf_model")

    # Log the model plots
    wandb.sklearn.plot_learning_curve(model,train_x,train_y)
    wandb.sklearn.plot_feature_importances(model)

    wandb.log({"accuracy":accuracy,
                "precision":precision,
                "recall":recall,
                "f1score":f1score, 
                "max_depth":max_depth,
                "n_estimators":n_estimators})

    wandb.sklearn.plot_classifier(model, 
                              train_x, test_x,
                              train_y, test_y,
                              y_pred, y_probas,
                              is_binary=True, 
                              model_name='rf_model')

    wandb.finish()