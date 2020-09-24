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


if get_ipython().__class__.__name__ != 'ZMQInteractiveShell':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-processes", type=int, required=True)
    n_procs = parser.parse_args().n_processes
else:
    n_procs = multiprocessing.cpu_count()


# In[4]:


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s - %(message)s')


# In[5]:


numpy.random.seed(42)


# In[6]:


TASKS = [9983, 9952, 3899, 219, 3954, 14964, 32, 6, 3510, 40, 9950, 53, 3512, 12, 3962, 39, 3577, 145682, 3794, 146824]


# In[7]:


def load_openml_task(task_id):
    task = openml.tasks.get_task(task_id)
    X, y = task.get_X_and_y("dataframe")
    
    X_enc = sklearn.preprocessing.OrdinalEncoder().fit_transform(X.values)
    X = pandas.DataFrame(X_enc, columns=X.columns, index=X.index)
    
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


# In[10]:


MODELS = {
    "rf": sklearn.ensemble.RandomForestClassifier(),
    "svm": sklearn.svm.SVC(probability=True),
    "logreg": sklearn.linear_model.LogisticRegression(max_iter=500),
    "nb": sklearn.naive_bayes.GaussianNB(),
    "mlp": sklearn.neural_network.MLPClassifier(max_iter=500)
}


# In[11]:


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


# In[12]:


def get_cv_metrics_for_model_and_task(model_id, task_id, pool, n_repeats, counter, start_at):
    
    X, y, splitter = load_openml_task(task_id) # repeated runs will use cached data
    
    promises = []
    for i, (train_idx, test_idx) in enumerate(splitter.split()):
        for j in range(n_repeats):
            counter += 1
            if counter < start_at:
                continue
            
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
        
    return promises, counter


# In[13]:


with multiprocessing.Pool(processes=n_procs) as pool:
    
    start_at = 0
    
    output_file = f"metrics_{int(time.time())}.dat"
    logging.info(f"Output to {output_file}")
    
    promises = []
    counter = 0
    for model_id in MODELS.keys():
        for task_id in TASKS:
            tmp, counter = get_cv_metrics_for_model_and_task(model_id, task_id, pool, 1, counter, start_at)
            promises.extend(tmp)
            
    logging.info(f"{len(promises)} promises submitted to pool")
            
    data = []
    for promise in promises:
        data.append(promise.get())
        logging.info(f"Finished promises: {len(data)}/{len(promises)} ({len(data)/len(promises)*100:.2f}%)")
        df = pandas.DataFrame(data)
        dump(df, output_file)


# In[ ]:


if get_ipython().__class__.__name__ != 'ZMQInteractiveShell':
    exit()


# In[ ]:


df = load("/home/maximl/Data/Experiment_data/results/riverrel/metrics_1600940535.dat")


# In[ ]:


df = load("metrics_1600957213.dat")


# In[ ]:


grouped_df = df.groupby(["model_id", "task_id", "repeat"]).aggregate("mean").drop(columns=["fold"]).reset_index()


# In[ ]:


grouped_df


# In[ ]:


def get_longform(df, cols=None, subject_cols=None):
    dfs = []
    
    if cols is None:
        cols = df.columns
    
    for col in cols:
        tmp_df = pandas.DataFrame(dict(        
            value=df[col], 
            metric=col,   
        ))
        for col2 in set(df.columns) - set(cols):
            tmp_df[col2] = df[col2]
            
        if subject_cols is not None:
            tmp_df["subject"] = df[subject_cols].apply(lambda row: '_'.join(row.values.astype(str)), axis=1)
            
        dfs.append(tmp_df)
        
    return pandas.concat(dfs)


# In[ ]:


long_df = get_longform(grouped_df, grouped_df.columns[3:], ["model_id", "task_id"])


# In[ ]:


long_df.shape


# In[ ]:


long_df.head()


# In[ ]:


seaborn.catplot(data=long_df[long_df["metric"].isin(["accuracy", "balanced_accuracy", "f1"])], x="model_id", y="value", col="metric", kind="box")


# In[ ]:


seaborn.catplot(data=long_df[long_df["metric"].isin(["accuracy", "balanced_accuracy", "f1"])], x="task_id", y="value", col="metric", kind="box")


# In[ ]:


seaborn.displot(data=long_df[long_df["metric"].isin(["ece", "ece_balanced", "peace"])], x="value", col="metric", rug=True, kind="kde")


# In[ ]:


seaborn.boxplot(data=long_df[long_df["metric"].isin(["ece", "ece_balanced", "peace"])], y="value", x="metric")


# In[ ]:


seaborn.lineplot(data=long_df[long_df["metric"].isin(["ece", "ece_balanced", "peace"])], y="value", x="metric", hue="subject", err_style="bars", palette="tab10")


# In[ ]:


import scipy.stats
import scikit_posthocs as sp


# In[ ]:


data = grouped_df.loc[grouped_df["repeat"] == 0, ["ece", "ece_balanced", "peace"]].values
scipy.stats.friedmanchisquare(data[0], data[1], data[2])


# In[ ]:


long_data = get_longform(grouped_df.loc[grouped_df["repeat"] == 0, ["ece", "ece_balanced", "peace"]])
sp.posthoc_conover(long_data, val_col="value", group_col="metric", p_adjust="holm")


# In[ ]:


for idx, model_df in grouped_df.groupby("model_id"):
    data = model_df.loc[:, ["ece", "ece_balanced", "peace"]]
    test = scipy.stats.friedmanchisquare(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2])
    print(idx)
    print(test)
    if test.pvalue < 0.05:
        long_data = get_longform(data)
        print(sp.posthoc_conover(long_data, val_col="value", group_col="metric", p_adjust="holm"))


# In[ ]:


seaborn.catplot(data=long_df[long_df["metric"].isin(["ece", "ece_balanced", "peace"])], x="metric", y="value", col="model_id", kind="box")

