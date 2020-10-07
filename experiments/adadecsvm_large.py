#!/usr/bin/env python
# coding: utf-8

# In[13]:


import pandas
import numpy
import seaborn
import matplotlib.pyplot as plt
import tabulate

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
import xgboost

from ridgereliability import plots, metrics

import os
import multiprocessing
import logging
import time
from joblib import load, dump


# In[14]:


def is_notebook():
    try:
        from IPython import get_ipython
        return get_ipython().__class__.__name__ == 'ZMQInteractiveShell'
    except ModuleNotFoundError:
        return False


# In[15]:


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


# In[16]:


logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(name)s %(asctime)s - %(message)s')
logging.captureWarnings(True)
logging.getLogger("openml").setLevel(logging.ERROR)

class NoRequestFilter(logging.Filter):
    def filter(self, record):
        return not "request for the URL" in record.getMessage()

logging.getLogger("root").addFilter(NoRequestFilter())
logging.getLogger().addFilter(NoRequestFilter())


# In[17]:


numpy.random.seed(42)


# In[18]:


def find_random_task(selected_tasks):
    tries = 0
    while True:
        df = openml.tasks.list_tasks(task_type_id=1, offset=tries*1000, output_format="dataframe", size=1000, status="active", number_missing_values=0)
        tries += 1
        if "NumberOfInstances" in df:
            df = df[(df["NumberOfInstances"] > 2000) & (df["NumberOfInstances"] < 10000)]
            df = df[~df["tid"].isin(list(selected_tasks))]
            if len(df) > 0:
                task = df.sample(n=1).iloc[0]["tid"]
                selected_tasks.append(task)
                return task


# In[19]:


LARGE_TASKS = [  3919,   4230,   3601,     30,   3891,   4190,   1792,     43,
         1908,   3786,   2939,   3488,   3843,   3603,    266,   4193,
         3528,   3842,   3525,   3777,   1923,   1809,    275,   3638,
         3684,   1800,   1916,   1794,   2074,     36,   3672,   2120,
          288,   3594,   3712,   3618,   3586,   4229,   3917,   1925,
         3816,   1767,   3668,   1883,   3698,   3907,   4233,   3485,
         3531,   3627,   1822,   3524,   3821,   3681,   3892,   3950,
          233,    260,   3894,   4236,   3884,   3735,   4215,   2121,
          258,   3481,   3839,   3510,     58,   1910,   3591,     28,
          273,     45,      3,   1938,   1807,   4186, 126022,  14969,
       145945,  75100,   4308, 145855,   9898,   4612,  75203,  75216,
       145891, 145985,   9952,   9921,  75169, 145903,  75242,  75124,
        75138,  10091]


# In[20]:


SMALL_TASKS = [4240, 4245, 1780,  248, 1896, 3891, 3934, 3611, 3837, 3763, 1784,
       2938,   12,  242, 3918, 3497, 3524, 3614, 3523,  246, 1923, 3925,
        277,   31, 3603, 3653, 3894,   52, 3058,  250, 4196, 3994,    3,
       3584,  252, 4200, 3527, 1787, 3778, 3993, 3841, 3959, 3996, 3846,
       4190, 3933,  275, 1927, 4234, 4238, 3997, 3520, 1892, 4241, 1811,
        266, 3521, 1806, 3635, 1916, 3730, 3806, 3710, 3892, 3657, 3775,
        270, 3491, 3716, 3753, 3676, 3788, 3538, 4235, 3585, 3731, 3702,
       3917, 3821,   45, 3689, 3598,  233, 3995, 2104, 3519, 3735, 1903,
       3777, 4243, 3828, 3733, 3056, 3583, 4225, 3695, 4198, 3617,  261]


# In[21]:


if random_tasks > 0:
    TASKS = random_tasks
else:
    TASKS = LARGE_TASKS


# In[22]:


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
        except Exception as e:
            if task_id is not None:
                raise e


# In[23]:


MODELS = {
    "adaboost": sklearn.ensemble.AdaBoostClassifier(n_estimators=500),
    "dectree": sklearn.tree.DecisionTreeClassifier(),
    "svm": sklearn.svm.LinearSVC()
}


# In[24]:


def fit_and_predict(model_id, Xt, yt, Xv, yv):
    # get and fit fresh model
    model = sklearn.base.clone(MODELS[model_id])
    model.fit(Xt, yt)

    # predict on test
    y_probs = model.predict_proba(Xv)
    y_preds = model.predict(Xv)
    
    return y_probs, y_preds, yv


# In[25]:


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


# In[26]:


with multiprocessing.Pool(processes=n_procs) as pool:

    start_at = 0

    output_file = f"metrics_{int(time.time())}.dat"
    logging.info(f"Output to {output_file}")

    counter = 0
    
    if type(TASKS) is int:
        iter_tasks = []
        tmp_tasks = []
        for i in range(TASKS):
            _, _, _, task = load_openml_task(selected_tasks=tmp_tasks)
            iter_tasks.append(task)
    else:
        iter_tasks = TASKS
        
    logging.info(f"Tasks: {iter_tasks}")
    
    data = []
    for model_id in MODELS.keys():
        
        logging.info(f"Start submitting promises for {model_id}")
        
        promises = []
        for task_id in iter_tasks:
            row, tmp, counter = get_cv_metrics_for_model_and_task(model_id, task_id, pool, counter, start_at, [])
            promises.append((row, tmp))
            logging.info(f"{len(promises)} tasks submitted to pool ({model_id})")
        logging.info(f"All {len(promises)} tasks submitted to pool ({model_id})")

        for i, (row, promise) in enumerate(promises):
            try:
                y_probs, y_preds, y_test = [], [], []
                for x in promise:
                    x = x.get()
                    y_probs.extend(x[0])
                    y_preds.extend(x[1])
                    y_test.extend(x[2])

                # stack fold results and compute metrics
                y_probs = numpy.array(y_probs)
                logging.debug(y_probs.shape)
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

                logging.info(f"Finished tasks: {i+1}/{len(promises)} ({(i+1)/len(promises)*100:.2f}%)")
            except Exception:
                logging.exception("Exception when collecting results")


# In[ ]:


if not is_notebook():
    exit()


# In[ ]:


tmp_df = load("/home/maximl/Data/Experiment_data/results/riverrel/datasets/random_openml/metrics_1601995877.dat")
tmp_df["model_id"] = "bagged_" + tmp_df["model_id"]


# In[ ]:


df = pandas.concat([
#     load("/home/maximl/Data/Experiment_data/results/riverrel/datasets/random_openml/metrics_1601494658.dat"),
    load("/home/maximl/Data/Experiment_data/results/riverrel/datasets/random_openml/metrics_1602009416.dat"),
])


# In[ ]:


df.columns = ["model_id", "task_id", "Accuracy", "Balanced Accuracy", "F1", "ECE", "Balanced ECE", "PEACE", "cw-ECE", "cw-PEACE"]


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


long_df = get_longform(df, df.columns[2:], ["model_id", "task_id"])


# In[ ]:


long_df.shape


# In[ ]:


long_df.head()


# In[ ]:


seaborn.set_theme("paper", "whitegrid", font_scale=1.5)


# In[ ]:


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
#     ["AdaBoost", "Decision Tree", "Logistic Regression", "Multi-layer Perceptron", "Gaussian Naive Bayes", "Random Forest", "SVM"],
    rotation=-30, ha="left")

plt.legend(handles[:3], labels[:3], bbox_to_anchor=(0., 1.02, 1., .102), loc='lower right',
           ncol=3, borderaxespad=0.)
plt.savefig("performance.pdf", bbox_inches="tight")


# In[ ]:


cols = ["ECE", "Balanced ECE", "PEACE"]


# In[ ]:


seaborn.displot(data=long_df[long_df["metric"].isin(cols)], x="value", hue="metric", rug=True, kind="kde")


# In[ ]:


g = seaborn.violinplot(data=long_df[long_df["metric"].isin(cols)], y="value", x="metric")
seaborn.stripplot(data=long_df[long_df["metric"].isin(cols)], x="metric", y="value", color=".25", s=2, alpha=.8)
g.set_xlabel("")
g.set_ylabel("Metric value")
# plt.savefig("metrics.pdf")


# In[ ]:


(df["PEACE"] - df["ECE"]).mean()


# In[ ]:


import scipy.stats
import scikit_posthocs as sp


# In[ ]:


df.head()


# In[ ]:


def map_stars(p):
    if p < 0.001:
        return "*"*3
    elif p < 0.01:
        return "*"*2
    elif p < 0.05:
        return "*"
    else:
        return ""


# In[ ]:


grid = seaborn.FacetGrid(data=df, col="model_id", col_wrap=4)
table_data = [["PEACE - ECE"], ["PEACE - Balanced ECE"], ["Balanced ECE - ECE"]]
for ax, (idx, model_df) in zip(grid.axes, df.groupby("model_id")):    
    
    diff_df = pandas.concat([
        pandas.DataFrame({"value": model_df["PEACE"] - model_df["ECE"], "diff": "PEACE - ECE"}),
        pandas.DataFrame({"value": model_df["PEACE"] - model_df["Balanced ECE"], "diff": "PEACE - Balanced ECE"}),
        pandas.DataFrame({"value": model_df["Balanced ECE"] - model_df["ECE"], "diff": "Balanced ECE - ECE"})
    ])
    
    data = model_df.loc[:, cols + ["task_id"]]
    test = scipy.stats.friedmanchisquare(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2])
    if test.pvalue < 0.05:
        a = sp.posthoc_conover_friedman(data.iloc[:, :3], p_adjust="bonferroni")
        colors = [
            "red" if a["PEACE"]["ECE"] < 0.05 else "grey",
            "red" if a["PEACE"]["Balanced ECE"] < 0.05 else "grey",
            "red" if a["Balanced ECE"]["ECE"] < 0.05 else "grey"
        ]
    else:
        colors = ["grey"]*3

    tmp_df = diff_df.groupby("diff").aggregate("mean")
    for i, (col1, col2) in enumerate([["PEACE", "ECE"], ["PEACE", "Balanced ECE"], ["Balanced ECE", "ECE"]]):
        table_data[i].append("$%+.3f$ %s" % (tmp_df.loc[f"{col1} - {col2}"], map_stars(a[col1][col2])))
    
    ax.set_title(idx)
    seaborn.boxplot(data=diff_df, y="diff", x="value", orient="h", ax=ax, palette=seaborn.color_palette(colors))
    ax.set_xlim(-0.05, 0.05)

grid.set_axis_labels("Difference", "Comparison")
plt.savefig("pairwise_comparisons.pdf", bbox_inches="tight")


# In[ ]:


grid = seaborn.FacetGrid(data=df1, col="model_id", col_wrap=4)
table_data = [["PEACE - ECE"], ["PEACE - Balanced ECE"], ["Balanced ECE - ECE"]]
for ax, (idx, model_df) in zip(grid.axes, df1.groupby("model_id")):    
    
    diff_df = pandas.concat([
        pandas.DataFrame({"value": model_df["PEACE"] - model_df["ECE"], "diff": "PEACE - ECE"}),
        pandas.DataFrame({"value": model_df["PEACE"] - model_df["Balanced ECE"], "diff": "PEACE - Balanced ECE"}),
        pandas.DataFrame({"value": model_df["Balanced ECE"] - model_df["ECE"], "diff": "Balanced ECE - ECE"})
    ])
    
    data = model_df.loc[:, cols + ["task_id"]]
    test = scipy.stats.friedmanchisquare(data.iloc[:, 0], data.iloc[:, 1], data.iloc[:, 2])
    if test.pvalue < 0.05:
        a = sp.posthoc_conover_friedman(data.iloc[:, :3], p_adjust="bonferroni")
        colors = [
            "red" if a["PEACE"]["ECE"] < 0.05 else "grey",
            "red" if a["PEACE"]["Balanced ECE"] < 0.05 else "grey",
            "red" if a["Balanced ECE"]["ECE"] < 0.05 else "grey"
        ]
    else:
        colors = ["grey"]*3

    tmp_df = diff_df.groupby("diff").aggregate("mean")
    for i, (col1, col2) in enumerate([["PEACE", "ECE"], ["PEACE", "Balanced ECE"], ["Balanced ECE", "ECE"]]):
        table_data[i].append("$%+.3f$ %s" % (tmp_df.loc[f"{col1} - {col2}"], map_stars(a[col1][col2])))
    
    ax.set_title(idx)
    seaborn.boxplot(data=diff_df, y="diff", x="value", orient="h", ax=ax, palette=seaborn.color_palette(colors))
    ax.set_xlim(-0.05, 0.05)

grid.set_axis_labels("Difference", "Comparison")
plt.savefig("pairwise_comparisons.pdf", bbox_inches="tight")


# In[ ]:


headers = ["AdaBoost", "DecTree", "LogReg", "MLP", "GNB", "RF", "SVM"]
print(tabulate.tabulate(table_data, headers=headers, tablefmt="latex").replace("\\$", "$"))


# In[ ]:


seaborn.catplot(data=long_df[long_df["metric"].isin(cols)], x="metric", y="value", col="model_id", kind="violin")


# In[ ]:


def get_task_meta(task_id):
    task = openml.tasks.get_task(task_id)
    d = task.get_dataset()
    return {
        "Task ID": task_id,
        "# classes": d.qualities["NumberOfClasses"],
        "# features":  d.qualities["NumberOfFeatures"],
        "# instances": d.qualities["NumberOfInstances"],
        "Dataset ID": d.dataset_id,
        "Dataset name": d.name,
        "Dataset URL": d.openml_url,
        "Dataset version": d.version,
        "Dataset creator": ", ".join(d.creator) if type(d.creator) is list else d.creator
    }


# In[ ]:


with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    tasks = pool.map(get_task_meta, df["task_id"].unique())


# In[ ]:


tasks = pandas.DataFrame(tasks)
tasks.head()


# In[ ]:


with open("task_table.tex", "w") as fp:
    fp.write(tabulate.tabulate(
        tasks[["Task ID", "Dataset ID", "Dataset name", "Dataset version", "# classes", "# features", "# instances"]].values, 
        headers=["Task ID", "Dataset ID", "Dataset name", "Dataset version", "# classes", "# features", "# instances"],
        tablefmt="latex"))


# In[ ]:


table = tasks.merge(df, right_on="task_id", left_on="Task ID")


# In[ ]:


tasks["# instances"].describe()


# In[ ]:


table["PEACE-ECE"] = table["PEACE"] - table["ECE"]


# In[ ]:


table["PEACE-balECE"] = table["PEACE"] - table["Balanced ECE"]
table["PEACE>=balECE"] = table["PEACE"] >= table["Balanced ECE"]


# In[ ]:


selector = table["model_id"] == "mlp"


# In[ ]:


fig, ax = plt.subplots(1, 3, figsize=(10, 4), constrained_layout=True)
seaborn.regplot(data=table, x="# instances", y="PEACE-balECE", ax=ax[0])
seaborn.regplot(data=table, x="# features", y="PEACE-balECE", ax=ax[1])
seaborn.regplot(data=table, x="# classes", y="PEACE-balECE", ax=ax[2])


# In[ ]:


import statsmodels.api as sm


# In[ ]:


X = pandas.concat([pandas.get_dummies(table["Task ID"]), pandas.get_dummies(table["model_id"]), table[["# instances", "# features", "# classes"]]], axis=1)


# In[ ]:


model = sm.RLM(table["PEACE-balECE"], sm.add_constant(X))
res = model.fit()


# In[ ]:


res.summary()


# In[ ]:


from functools import partial


# In[ ]:


row = grouped_df.iloc[(grouped_df["PEACE"] - grouped_df["ECE"]).sort_values().index[0]]


# In[ ]:


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


# In[ ]:


ada_data = func(sklearn.ensemble.AdaBoostClassifier)


# In[ ]:


logreg_data = func(partial(sklearn.linear_model.LogisticRegression, max_iter=1000))


# In[ ]:


svm_data = func(partial(sklearn.svm.SVC, max_iter=1000, probability=True))


# In[ ]:


mlp_data = func(partial(sklearn.neural_network.MLPClassifier, max_iter=1000))


# In[ ]:


rf_data = func(sklearn.ensemble.RandomForestClassifier)


# In[ ]:


fig, axes = plt.subplots(1, 5, sharex=True, sharey=True, figsize=(20, 4))

for ax, data in zip(axes, [ada_data, logreg_data, svm_data, mlp_data, rf_data]):
    y_probs = numpy.hstack([p.max(axis=1) for p in data[0]])
    y_preds = numpy.hstack(data[1])
    y_test = numpy.hstack(data[2])
    ax.set_title(f"ECE: {metrics.ece(y_probs, y_preds, y_test, bins=15):.3f}, PEACE: {metrics.peace(y_probs, y_preds, y_test, bins=15):.3f}")

    seaborn.histplot(y_probs, ax=ax, bins=numpy.histogram_bin_edges(y_probs, bins=15, range=(0,1)))


# In[ ]:


plots.river_reliability_diagram(mlp_data[0][i].max(axis=1), mlp_data[1][i], mlp_data[2][i], bins=15)


# In[ ]:


plots.river_reliability_diagram(logreg_data[0][i].max(axis=1), logreg_data[1][i], logreg_data[2][i], bins=15)


# In[ ]:


plots.confidence_reliability_diagram(ada_data[0][i].max(axis=1), ada_data[1][i], ada_data[2][i], bins=15)


# In[ ]:


plots.confidence_reliability_diagram(logreg_data[0][i].max(axis=1), logreg_data[1][i], logreg_data[2][i], bins=15)


# In[ ]:




