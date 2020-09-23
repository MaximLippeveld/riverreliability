#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt

import openml

import sklearn.metrics
import sklearn.ensemble
import sklearn.svm
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.base
import sklearn.utils
import sklearn.linear_model
import sklearn.naive_bayes
import sklearn.neural_network

from ridgereliability import plots, metrics

import os
import multiprocessing
import logging
import time
from joblib import load, dump


# In[3]:


logging.basicConfig(level=logging.INFO)


# In[4]:


numpy.random.seed(42)


# In[5]:


TASKS = [9983, 9952, 3899, 219, 3954, 14964, 32, 6, 3510, 40]


# In[6]:


def load_openml_task(task_id):
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y("dataframe")
    n_repeats, n_folds, n_samples = task.get_split_dimensions()

    folds = numpy.empty((len(X)), dtype=int)
    for fold_idx in range(n_folds):
        _, test_indices = task.get_train_test_split_indices(
            repeat=0,
            fold=fold_idx,
            sample=0,
        )
        
        folds[test_indices] = fold_idx
        
    splitter = sklearn.model_selection.PredefinedSplit(folds)
                
    return X, y, splitter        


# In[9]:


MODELS = {
    "rf": sklearn.ensemble.RandomForestClassifier(),
    "svm": sklearn.svm.SVC(probability=True),
    "logreg": sklearn.linear_model.LogisticRegression(),
    "nb": sklearn.naive_bayes.GaussianNB(),
    "mlp": sklearn.neural_network.MLPClassifier()
}


# In[10]:


def get_fold_metrics_for_model(row, Xt, yt, Xv, yv):

    # get and fit fresh model
    model = sklearn.base.clone(MODELS[row["model_id"]])
    model.fit(Xt, yt)

    # compute metrics on test data
    y_probs = model.predict_proba(Xv)
    y_probs_max = y_probs.max(axis=1)
    y_preds = model.predict(Xv)
    y_test = yv

    bins = 15
    row.update({
        "accuracy": sklearn.metrics.accuracy_score(y_test, y_preds),
        "balanced_accuracy": sklearn.metrics.balanced_accuracy_score(y_test, y_preds),
        "f1": sklearn.metrics.f1_score(y_test, y_preds, average="weighted"),
        'ece': metrics.ece(y_probs_max, y_preds, y_test, bins=bins),
        'ece_balanced': metrics.ece(y_probs_max, y_preds, y_test, balanced=True, bins=bins),
        'peace': metrics.peace(y_probs_max, y_preds, y_test, bins=bins),
        'class_wise_ece': metrics.class_wise_error(y_probs, y_preds, y_test, metrics.ece, bins=bins),
        'class_wise_peace': metrics.class_wise_error(y_probs, y_preds, y_test, metrics.peace, bins=bins)
    })
    
    return row


# In[11]:


def get_cv_metrics_for_model_and_task(model_id, task_id, pool, n_repeats):
    
    X, y, splitter = load_openml_task(task_id) # repeated runs will use cached data
    
    promises = []
    for i, (train_idx, test_idx) in enumerate(splitter.split()):
        for j in range(n_repeats):
            row = {
                "fold": i,
                "repeat": j,
                "model_id": model_id,
                "task_id": task_id,
            }

            # split data
            Xt, yt = X.iloc[train_idx], y.iloc[train_idx]
            Xv, yv = X.iloc[test_idx], y.iloc[test_idx]

            promise = pool.apply_async(
                get_fold_metrics_for_model,
                (row, Xt, yt, Xv, yv)
            )
            promises.append(promise)
        
    return promises


# In[12]:


with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    
    output_file = f"metrics_{int(time.time())}.dat"
    logging.info(f"Output to {output_file}")
    
    promises = []
    for model_id in MODELS.keys():
        for task_id in TASKS:
            promises.extend(get_cv_metrics_for_model_and_task(model_id, task_id, pool, 1))
            
    data = []
    for promise in promises:
        data.append(promise.get())
        logging.info(f"Finished promises: {len(data)}/{len(promises)} ({len(data)/len(promises)*100:.2f}%)")
        df = pandas.DataFrame(data)
        dump(df, output_file)


# In[ ]:


exit()


# In[ ]:


grouped_df = df.groupby(["model_id", "task_id", "repeat"]).aggregate("mean").drop(columns=["fold"]).reset_index()


# In[ ]:


grouped_df


# In[ ]:


dfs = []
for col in grouped_df.iloc[:, 3:]:
    dfs.append(
        pandas.DataFrame(
            dict(value=grouped_df[col], metric=col, subject=grouped_df["model_id"] + grouped_df["task_id"].astype(str))
        )
    )
long_df = pandas.concat(dfs)


# In[ ]:


long_df


# In[ ]:


seaborn.displot(data=long_df[long_df["metric"].isin(["ece", "ece_balanced", "peace"])], x="value", col="metric", rug=True, kind="kde")


# In[ ]:


seaborn.boxplot(data=long_df[long_df["metric"].isin(["ece", "ece_balanced", "peace"])], y="value", x="metric")


# In[ ]:


seaborn.lineplot(data=long_df[long_df["metric"].isin(["ece", "ece_balanced", "peace"])], y="value", x="metric", hue="subject", err_style="bars", palette="tab10")


# In[ ]:




