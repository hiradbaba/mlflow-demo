import sklearn
import os
import numpy as np
from numpy.random import shuffle
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, f1_score
import mlflow

def get_data():
    df = pd.read_csv('a1/iris.csv')
    X = df.to_numpy()
    shuffle(X)
    encoder = OrdinalEncoder()
    Xt = encoder.fit_transform(X)
    s = int(len(Xt) * 0.8)
    x_train, x_test, y_train, y_test = Xt[:s,:-1], Xt[s:,:-1], Xt[:s,-1], Xt[s:,-1]

    train_df = pd.DataFrame(columns=df.columns, data=Xt[:s])
    test_df = pd.DataFrame(columns=df.columns, data=Xt[s:])

    train_df.to_csv('a1/train.csv', index=False)
    test_df.to_csv('a1/test.csv', index=False)

    return x_train, x_test, y_train, y_test

def log_and_clean_data():
    mlflow.log_artifact('a1/train.csv')
    mlflow.log_artifact('a1/test.csv')
    os.remove('a1/train.csv')
    os.remove('a1/test.csv')

def train_model(x, y, **kwargs):
    lr = LogisticRegression(**kwargs)
    lr.fit(x,y)
    return lr

def get_metrics(lr, x, y):
    y_pred = lr.predict(x)
    acc = accuracy_score(y, y_pred)
    recall = recall_score(y, y_pred, average='micro')
    f1 = f1_score(y, y_pred, average='micro')
    return acc, recall, f1

def get_parameters(model_index):
    if model_index == 0:
        return [
            {"random_state": 42, "solver": 'lgbfs', "fit_intercept": False},
            {"random_state": 42, "solver": 'lgbfs', "fit_intercept": True},
        ]
    elif model_index == 1:
        return [
            {"random_state": 42, "solver": 'saga', "fit_intercept": False},
            {"random_state": 42, "solver": 'saga', "fit_intercept": True},
        ]

for i in range(2):
    exp = mlflow.set_experiment(f"Exp-{i+1}")
    run = 1
    param_set = get_parameters(i)
    for params in param_set:
        with mlflow.start_run(experiment_id=exp.experiment_id, run_name=f'run-1.{run}'):
            mlflow.log_params(params)
            x_train, x_test, y_train, y_test = get_data()
            log_and_clean_data()
            lr = train_model(x_train,y_train)
            acc_train, recall_train, f1_train = get_metrics(lr, x_train, y_train)
            print(f"Accuracy: {acc_train}")
            print(f"Recall: {recall_train}")
            print(f"F1: {f1_train}")
            mlflow.log_metrics({
                "Train Accuracy": acc_train,
                "Train Recall": recall_train,
                "Train F1": f1_train
            })

            acc_test, recall_test, f1_test = get_metrics(lr, x_train, y_train)
            print(f"Accuracy: {acc_test}")
            print(f"Recall: {recall_test}")
            print(f"F1: {f1_test}")
            mlflow.log_metrics({
                "Test Accuracy": acc_test,
                "Test Recall": recall_test,
                "Test F1": f1_test
            })
            mlflow.sklearn.log_model(lr, f"model-{i}-{run}")
            mlflow.set_tags({
                "engineering": "ML Platform",
                "release.candidate": f"RC-{i}.{run}",
                "release.version": "0.1.0"}
                )
            run += 1
lar = mlflow.last_active_run()
print(f"Last Active Run: {lar.info.run_name}-{lar.info.run_id}")

            

