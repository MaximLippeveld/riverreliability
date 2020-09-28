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
import sklearn.tree

from ridgereliability import plots, metrics

import os
import multiprocessing
import logging
import time
from joblib import load, dump


# In[3]:


def is_notebook():
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except ModuleNotFoundError:
        return False


# In[4]:


if is_notebook():
    n_procs = multiprocessing.cpu_count()//2
    random_tasks = 2
else:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-processes", type=int, required=True)
    parser.add_argument("--random-tasks", type=int, default=0)
    args = parser.parse_args()
    n_procs = args.n_processes
    random_tasks = args.random_tasks


# In[5]:


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(asctime)s - %(message)s')
logging.captureWarnings(True)


# In[6]:


numpy.random.seed(42)


# In[6]:


def find_random_task(offset):
    while True:
        df = openml.tasks.list_tasks(task_type_id=1, offset=offset, output_format="dataframe", size=10000, status="active", number_missing_values=0)
        if "NumberOfInstances" in df:
            df = df[(df["NumberOfInstances"] > 100) & (df["NumberOfInstances"] < 1000)]
            if len(df) > 0:
                return df.sample(n=1).iloc[0]["tid"]


# In[7]:


if random_tasks > 0:
    TASKS = random_tasks
else:
    TASKS = [9983, 9952, 3899, 219, 3954, 14964, 32, 6, 3510, 40, 9950, 53, 3512, 12, 3962, 39, 3577, 145682, 3794, 146824]


# In[8]:


def load_openml_task(task_id=None, offset=0):
    
    while True:
        if task_id is None:
            curr_id = find_random_task(offset)
        else:
            curr_id = task_id

        try: 
            task = openml.tasks.get_task(curr_id)
            X, y = task.get_X_and_y("array")
            X, y = sklearn.utils.indexable(X, y)
            
            target_type = sklearn.utils.multiclass.type_of_target(y)
            if target_type not in ["binary", "multiclass"]:
                continue

            if hasattr(X, "toarray"):
                X = X.toarray()
            if hasattr(y, "toarray"):
                y = y.toarray()

            X = sklearn.preprocessing.OrdinalEncoder().fit_transform(X)

            n_repeats, n_folds, n_samples = task.get_split_dimensions()
            
            if n_folds > 10:
                continue

            folds = numpy.full((len(X)), dtype=int, fill_value=-1)
            for fold_idx in range(n_folds):
                _, test_indices = task.get_train_test_split_indices(
                    repeat=0,
                    fold=fold_idx,
                    sample=0,
                )

                folds[test_indices] = fold_idx

            splitter = sklearn.model_selection.PredefinedSplit(folds)

            return X, y, splitter, curr_id
        except:
            if task_id is not None:
                raise e


# In[11]:


MODELS = {
    "rf": sklearn.ensemble.RandomForestClassifier(),
    "svm": sklearn.svm.SVC(probability=True),
    "logreg": sklearn.linear_model.LogisticRegression(max_iter=1000),
    "nb": sklearn.naive_bayes.GaussianNB(),
    "mlp": sklearn.neural_network.MLPClassifier(max_iter=1000),
    "adaboost": sklearn.ensemble.AdaBoostClassifier(n_estimators=500),
    "dectree": sklearn.tree.DecisionTreeClassifier()
}


# In[12]:


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


# In[13]:


def get_cv_metrics_for_model_and_task(model_id, task_id, pool, n_repeats, counter, start_at):
    
    X, y, splitter, task_id = load_openml_task(task_id, offset=counter) # repeated runs will use cached data
    
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
            Xt, yt = X[train_idx], y[train_idx]
            Xv, yv = X[test_idx], y[test_idx]

            promise = pool.apply_async(
                get_fold_metrics_for_model,
                (row, Xt, yt, Xv, yv)
            )
            promises.append(promise)
    
    logging.info(f"Promises for single cv: {len(promises)}")
    return promises, counter


# In[14]:


with multiprocessing.Pool(processes=n_procs) as pool:

    start_at = 0

    output_file = f"metrics_{int(time.time())}.dat"
    logging.info(f"Output to {output_file}")

    promises = []
    counter = 0
    
    if type(TASKS) is int:
        iter_tasks = [None]*TASKS
    else:
        iter_tasks = TASKS
    
    for model_id in MODELS.keys():
        for task_id in iter_tasks:
            tmp, counter = get_cv_metrics_for_model_and_task(model_id, task_id, pool, 1, counter, start_at)
            promises.extend(tmp)
            logging.info(f"{len(promises)} promises submitted to pool ({model_id, task_id})")

    logging.info(f"{len(promises)} promises submitted to pool")

    data = []
    for promise in promises:
        data.append(promise.get())
        logging.info(f"Finished promises: {len(data)}/{len(promises)} ({len(data)/len(promises)*100:.2f}%)")
        df = pandas.DataFrame(data)
        dump(df, output_file)


# In[16]:


if not is_notebook():
    exit()


# In[7]:


df = load("/home/maximl/Data/Experiment_data/results/riverrel/datasets/random/metrics_1601149607.dat")


# In[11]:


df


# In[8]:


grouped_df = df.groupby(["model_id", "task_id", "repeat"]).aggregate("mean").drop(columns=["fold"]).reset_index()


# In[9]:


grouped_df


# In[20]:


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


# In[21]:


long_df = get_longform(grouped_df, grouped_df.columns[3:], ["model_id", "task_id"])


# In[22]:


long_df.shape


# In[23]:


long_df.head()


# In[24]:


seaborn.catplot(data=long_df[long_df["metric"].isin(["accuracy", "balanced_accuracy", "f1"])], x="model_id", y="value", col="metric", kind="box")


# In[25]:


seaborn.catplot(data=long_df[long_df["metric"].isin(["accuracy", "balanced_accuracy", "f1"])], x="task_id", y="value", col="metric", kind="box")


# In[26]:


seaborn.displot(data=long_df[long_df["metric"].isin(["ece", "ece_balanced", "peace"])], x="value", col="metric", rug=True, kind="kde")


# In[27]:


seaborn.boxenplot(data=long_df[long_df["metric"].isin(["ece", "ece_balanced", "peace"])], y="value", x="metric")


# In[28]:


long_df["metric_ord"] = long_df["metric"].map(lambda a: numpy.unique(long_df["metric"]).tolist().index(a))


# In[29]:


seaborn.lmplot(data=long_df[long_df["metric"].isin(["ece", "ece_balanced", "peace"])], y="value", x="metric_ord")


# In[30]:


import scipy.stats
import scikit_posthocs as sp


# In[31]:


data = grouped_df.loc[grouped_df["repeat"] == 0, ["ece", "ece_balanced", "peace"]].values
scipy.stats.friedmanchisquare(data[0], data[1], data[2])


# In[32]:


long_data = get_longform(grouped_df.loc[grouped_df["repeat"] == 0, ["ece", "ece_balanced", "peace"]])
sp.posthoc_conover(long_data, val_col="value", group_col="metric", p_adjust="holm")


# In[33]:


for idx, model_df in grouped_df.groupby("model_id"):
    data = model_df.loc[:, ["ece", "ece_balanced", "peace"]]
    test = scipy.stats.friedmanchisquare(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2])
    print(idx)
    print(test)
    if test.pvalue < 0.05:
        long_data = get_longform(data)
        print(sp.posthoc_conover(long_data, val_col="value", group_col="metric", p_adjust="holm"))


# In[34]:


seaborn.catplot(data=long_df[long_df["metric"].isin(["ece", "ece_balanced", "peace"])], x="metric", y="value", col="model_id", kind="box")

