import numpy as np
import pandas as pd
import json
import pickle
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(cm):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


def calculate_accuracy(df_true, df_pred):
    total_matches = (df_true == df_pred).sum().sum()
    total_elements = df_true.size
    accuracy = total_matches / total_elements
    return accuracy


data_train = pd.read_csv("../baseline_dataset/0to20k_wa_baseline_train.csv")
data_test = pd.read_csv("../baseline_dataset/0to20k_wa_baseline_test.csv")
data_all = pd.concat([data_train, data_test], axis=0)

targets = ["TOT_INJ", "SEVERITY", "ACCTYPE"]

for target in targets:
    with open("DAG/" + target + "_DAG.pkl", "rb") as f:
        DAG = pickle.load(f)
    model = BayesianNetwork(DAG.edges())
    t_data_test = data_test.loc[:, list(model.nodes())]
    t_data_train = data_train.loc[:, list(model.nodes())]
    t_data_all = data_all.loc[:, list(model.nodes())]

    state_names = {}
    for key, _ in t_data_all.items():
        state_names[key] = list(set(t_data_all[key]))

    model.fit(
        t_data_train,
        # estimator=BayesianEstimator,
        n_jobs=-1,
        state_names=state_names,
    )

    r = model.predict(t_data_test.drop([target], axis=1))
    gt = t_data_test.loc[:, [target]].values.tolist()
    pre = r[target].tolist()
    cm = confusion_matrix(gt, pre)
    precision = precision_score(gt, pre, average="macro")
    recall = recall_score(gt, pre, average="macro")
    f1 = f1_score(gt, pre, average="macro")
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1 Score: ", f1)
    print("Accuracy: ", calculate_accuracy(t_data_test.loc[:, [target]], r))
    # plot_confusion_matrix(cm)
