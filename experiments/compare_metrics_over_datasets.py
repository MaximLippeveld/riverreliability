#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


def is_notebook():
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except ModuleNotFoundError:
        return False


# In[3]:


if is_notebook():
    n_procs = multiprocessing.cpu_count()//2
    random_tasks = 1
else:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-processes", type=int, required=True)
    parser.add_argument("--random-tasks", type=int, default=0)
    args = parser.parse_args()
    n_procs = args.n_processes
    random_tasks = args.random_tasks


# In[4]:


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(name)s %(asctime)s - %(message)s')
logging.captureWarnings(True)
logging.getLogger("openml").setLevel(logging.ERROR)

class NoRequestFilter(logging.Filter):
    def filter(self, record):
        return not "request for the URL" in record.getMessage()

logging.getLogger("root").addFilter(NoRequestFilter())
logging.getLogger().addFilter(NoRequestFilter())


# In[5]:


numpy.random.seed(42)


# In[6]:


def find_random_task(selected_tasks):
    tries = 0
    while True:
        df = openml.tasks.list_tasks(task_type_id=1, offset=tries*1000, output_format="dataframe", size=1000, status="active", number_missing_values=0)
        tries += 1
        if "NumberOfInstances" in df:
            df = df[(df["NumberOfInstances"] > 100) & (df["NumberOfInstances"] < 5000)]
            df = df[~df["tid"].isin(list(selected_tasks))]
            if len(df) > 0:
                task = df.sample(n=1).iloc[0]["tid"]
                selected_tasks.append(task)
                return task


# In[7]:


if random_tasks > 0:
    TASKS = random_tasks
else:
    TASKS = [9983, 9952, 3899, 219, 3954, 14964, 32, 6, 3510, 40, 9950, 53, 3512, 12, 3962, 39, 3577, 145682, 3794, 146824]


# In[8]:


def load_openml_task(task_id=None, selected_tasks=[]):
    
    while True:
        if task_id is None:
            curr_id = find_random_task(selected_tasks)
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


# In[17]:


MODELS = {
    "rf": sklearn.ensemble.RandomForestClassifier(),
    "svm": sklearn.svm.SVC(probability=True),
    "logreg": sklearn.linear_model.LogisticRegression(max_iter=1000),
    "nb": sklearn.naive_bayes.GaussianNB(),
    "mlp": sklearn.neural_network.MLPClassifier(max_iter=1000),
    "adaboost": sklearn.ensemble.AdaBoostClassifier(n_estimators=500),
    "dectree": sklearn.tree.DecisionTreeClassifier()
}


# In[11]:


def fit_and_predict(model_id, Xt, yt, Xv, yv):
    # get and fit fresh model
    model = sklearn.base.clone(MODELS[model_id])
    model.fit(Xt, yt)

    # predict on test
    y_probs = model.predict_proba(Xv)
    y_preds = model.predict(Xv)
    
    return y_probs, y_preds, yv


# In[12]:


def get_cv_metrics_for_model_and_task(model_id, task_id, pool, counter, start_at, selected_tasks):
    X, y, splitter, task_id = load_openml_task(task_id, selected_tasks=selected_tasks) # repeated runs will use cached data
    
    row = {
        "model_id": model_id,
        "task_id": task_id,
    }

    promises = []
    for i, (train_idx, test_idx) in enumerate(splitter.split()):
        counter += 1
        if counter < start_at:
            continue

        # split data
        Xt, yt = X[train_idx], y[train_idx]
        Xv, yv = X[test_idx], y[test_idx]

        promise = pool.apply_async(
            fit_and_predict,
            (model_id, Xt, yt, Xv, yv)
        )
        promises.append(promise)

    logging.info(f"Promises for single cv: {len(promises)}")
    return row, promises, counter


# In[18]:


with multiprocessing.Pool(processes=n_procs) as pool:
    
    manager = multiprocessing.Manager()
    selected_tasks = manager.list()

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
            row, tmp, counter = get_cv_metrics_for_model_and_task(model_id, task_id, pool, counter, start_at, selected_tasks)
            promises.append((row, tmp))
            logging.info(f"{len(promises)} tasks submitted to pool ({model_id})")

    logging.info(f"{len(promises)} tasks submitted to pool")

    data = []
    for row, promise in promises:
        try:
            y_probs, y_preds, y_test = [], [], []
            for x in promise:
                x = x.get()
                y_probs.extend(x[0])
                y_preds.extend(x[1])
                y_test.extend(x[2])
            
            # stack fold results and compute metrics
            y_probs = numpy.array(y_probs)
            y_probs_max = y_probs.max(axis=1)
            y_preds = numpy.array(y_preds)
            y_test = numpy.array(y_test)
                
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
            
            # update data and dump intermediate dataframe
            data.append(row)
            df = pandas.DataFrame(data)
            dump(df, output_file)
            
            logging.info(f"Finished tasks: {len(data)}/{len(promises)} ({len(data)/len(promises)*100:.2f}%)")
        except Exception:
            logging.exception("Exception when collecting results")


# In[ ]:


if not is_notebook():
    exit()


# In[12]:


df = load("/home/maximl/Data/Experiment_data/results/riverrel/datasets/random_openml/metrics_1601367377.dat")


# In[19]:


df = load("metrics_1601458186.dat")


# In[20]:


df.head()


# In[23]:


df.columns = ["model_id", "task_id", "Accuracy", "Balanced Accuracy", "F1", "ECE", "Balanced ECE", "PEACE", "cw-ECE", "cw-PEACE"]


# In[25]:


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


# In[26]:


long_df = get_longform(df, df.columns[2:], ["model_id", "task_id"])


# In[27]:


long_df.shape


# In[28]:


long_df.head()


# In[29]:


seaborn.set_theme("paper", "whitegrid", font_scale=1.5)


# In[30]:


fig, ax = plt.subplots(figsize=(10, 6))
seaborn.boxplot(
    data=long_df[long_df["metric"].isin(["Accuracy", "Balanced Accuracy", "F1"])], 
    x="model_id", y="value", hue="metric", saturation=.7, ax=ax
)
seaborn.stripplot(
    data=long_df[long_df["metric"].isin(["Accuracy", "Balanced Accuracy", "F1"])], 
    x="model_id", y="value", hue="metric", dodge=True, color=".25", s=3, alpha=.8, ax=ax
)
handles, labels = ax.get_legend_handles_labels()
ax.set_xlabel("Model")
ax.set_ylabel("Metric value")
ax.set_ylim(0, 1)
ax.set_xticklabels(
    ["AdaBoost", "Decision Tree", "Logistic Regression", "Multi-layer Perceptron", "Gaussian Naive Bayes", "Random Forest", "SVM"],
    rotation=-30, ha="left")

plt.legend(handles[:3], labels[:3], bbox_to_anchor=(0., 1.02, 1., .102), loc='lower right',
           ncol=3, borderaxespad=0.)
plt.savefig("performance.pdf", bbox_inches="tight")


# In[31]:


cols = ["ECE", "Balanced ECE", "PEACE"]


# In[32]:


seaborn.displot(data=long_df[long_df["metric"].isin(cols)], x="value", hue="metric", rug=True, kind="kde")


# In[34]:


g = seaborn.violinplot(data=long_df[long_df["metric"].isin(cols)], y="value", x="metric")
seaborn.stripplot(data=long_df[long_df["metric"].isin(cols)], x="metric", y="value", color=".25", s=2, alpha=.8)
g.set_xlabel("")
g.set_ylabel("Metric value")
# plt.savefig("metrics.pdf")


# In[35]:


long_df["metric_ord"] = long_df["metric"].map(lambda a: numpy.unique(long_df["metric"]).tolist().index(a))


# In[36]:


(df["PEACE"] - df["ECE"]).mean()


# In[21]:


import scipy.stats
import scikit_posthocs as sp


# In[22]:


data = grouped_df.loc[grouped_df["repeat"] == 0, cols].values
scipy.stats.friedmanchisquare(data[0], data[1], data[2])


# In[23]:


long_data = get_longform(grouped_df.loc[grouped_df["repeat"] == 0, cols])
sp.posthoc_conover(long_data, val_col="value", group_col="metric", p_adjust="holm")


# In[24]:


for idx, model_df in grouped_df.groupby("model_id"):
    data = model_df.loc[:, cols]
    test = scipy.stats.friedmanchisquare(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2])
    print(idx)
    print(test)
    
    if test.pvalue < 0.05:
        long_data = get_longform(data)
        print(sp.posthoc_conover(long_data, val_col="value", group_col="metric", p_adjust="holm"))
    print("-"*20)


# In[26]:


seaborn.catplot(data=long_df[long_df["metric"].isin(cols)], x="metric", y="value", col="model_id", kind="violin")


# In[91]:


def get_task_meta(task_id):
    task = openml.tasks.get_task(task_id)
    d = task.get_dataset()
    return dict(
        task_id = task_id,
        n_classes = d.qualities["NumberOfClasses"],
        n_features = d.qualities["NumberOfFeatures"],
        n_instances = d.qualities["NumberOfInstances"]
    )


# In[92]:


with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    tasks = pool.map(get_task_meta, grouped_df["task_id"])


# In[93]:


tasks_meta = pandas.DataFrame(tasks).set_index("task_id").join(grouped_df.set_index("task_id"))


# In[94]:


tasks_meta["PEACE-ECE"] = tasks_meta["PEACE"] - tasks_meta["ECE"]


# In[95]:


tasks_meta["PEACE>ECE"] = tasks_meta["PEACE"] > tasks_meta["ECE"]


# In[103]:


tasks_meta["PEACE>=ECE"] = tasks_meta["PEACE"] >= tasks_meta["ECE"]


# In[109]:


seaborn.displot(data=tasks_meta, x="n_instances", col="PEACE>=ECE", kind="hist")


# In[110]:


seaborn.displot(data=tasks_meta, x="n_classes", col="PEACE>=ECE", kind="hist")


# In[107]:


seaborn.displot(data=tasks_meta, x="n_features", hue="PEACE>=ECE", kind="kde")


# In[21]:


from functools import partial


# In[22]:


row = grouped_df.iloc[(grouped_df["PEACE"] - grouped_df["ECE"]).sort_values().index[0]]


# In[23]:


def func(model):
    X, y, splitter, task_id = load_openml_task(row["task_id"])

    y_probs = [[]]*splitter.get_n_splits()
    y_preds = [[]]*splitter.get_n_splits()
    y_test = [[]]*splitter.get_n_splits()

    for i, (train_idx, test_idx) in enumerate(splitter.split()):
        # split data
        Xt, yt = X[train_idx], y[train_idx]
        Xv, yv = X[test_idx], y[test_idx]

        # train adaboost
        model_instance = model()
        model_instance.fit(Xt, yt)

        y_probs[i] = model_instance.predict_proba(Xv)
        y_preds[i] = model_instance.predict(Xv)
        y_test[i] = yv
        
    return y_probs, y_preds, y_test


# In[24]:


ada_data = func(sklearn.ensemble.AdaBoostClassifier)


# In[25]:


logreg_data = func(partial(sklearn.linear_model.LogisticRegression, max_iter=1000))


# In[26]:


svm_data = func(partial(sklearn.svm.SVC, max_iter=1000, probability=True))


# In[27]:


mlp_data = func(partial(sklearn.neural_network.MLPClassifier, max_iter=1000))


# In[28]:


rf_data = func(sklearn.ensemble.RandomForestClassifier)


# In[30]:


fig, axes = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(20, 4))

for ax, data in zip(axes, [ada_data, logreg_data, svm_data, mlp_data, rf_data]):
    y_probs = numpy.hstack([p.max(axis=1) for p in data[0]])
    y_preds = numpy.hstack(data[1])
    y_test = numpy.hstack(data[2])
    ax.set_title(f"ECE: {metrics.ece(y_probs, y_preds, y_test, bins=15):.3f}, PEACE: {metrics.peace(y_probs, y_preds, y_test, bins=15):.3f}")

    seaborn.histplot(y_probs, ax=ax, bins=numpy.histogram_bin_edges(y_probs, bins=15, range=(0,1)))


# In[31]:


plots.river_reliability_diagram(mlp_data[0][i].max(axis=1), mlp_data[1][i], mlp_data[2][i], bins=15)


# In[45]:


plots.river_reliability_diagram(logreg_data[0][i].max(axis=1), logreg_data[1][i], logreg_data[2][i], bins=15)


# In[46]:


plots.confidence_reliability_diagram(ada_data[0][i].max(axis=1), ada_data[1][i], ada_data[2][i], bins=15)


# In[47]:


plots.confidence_reliability_diagram(logreg_data[0][i].max(axis=1), logreg_data[1][i], logreg_data[2][i], bins=15)


# In[ ]:




