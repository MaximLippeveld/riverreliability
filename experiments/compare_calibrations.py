#!/usr/bin/env python
# coding: utf-8

# In[2]:


import tensorflow as tf
import numpy as np
import pandas as pd
import os
from joblib import dump, load
import matplotlib.pyplot as plt
from glob import glob
from ridgereliability import metrics, plots
import sklearn.model_selection
import sklearn.metrics
from dirichlet.calib import tempscaling, vectorscaling
from dirichlet import DirichletCalibrator
from sklearn.base import clone
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.utils import check_X_y, indexable, column_or_1d
from sklearn.utils.validation import check_is_fitted


# In[3]:


import logging
logging.basicConfig(level=logging.INFO)


# In[4]:


from enum import Enum
class F(Enum):
    TRAIN = 1
    EVAL = 2


# In[5]:


f = F.TRAIN
epochs = 1


# In[6]:


if f is F.TRAIN:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str, required=True)
    output_dir = parser.parse_args().out
elif f is F.EVAL:
    output_dir = "models"


# In[ ]:


if f is F.TRAIN:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                print(gpu)
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
            
    dev = [d for d in tf.config.experimental.list_logical_devices() if d.device_type=="GPU"][0]


# In[ ]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

mean = x_train.mean()
std = x_train.std()

x_test = (x_test - mean) / (std + 1e-7)
y_test = tf.keras.utils.to_categorical(y_test)

x_train = (x_train - mean) / (std + 1e-7)
y_train = tf.keras.utils.to_categorical(y_train)


# In[ ]:


models = {}


# In[ ]:


if f is F.TRAIN:
    with tf.device(dev.name):
        model = tf.keras.applications.DenseNet121(
            include_top=True,
            weights=None,
            classes=10,
            input_shape=[32, 32, 3]
        )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=['accuracy'])

    model.save_weights(os.path.join(output_dir, 'model.h5'))


# In[ ]:


if f is F.TRAIN:
    with tf.device(dev.name):
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
                    featurewise_center=False,  # set input mean to 0 over the dataset
                    samplewise_center=False,  # set each sample mean to 0
                    featurewise_std_normalization=False,  # divide inputs by std of the dataset
                    samplewise_std_normalization=False,  # divide each input by its std
                    zca_whitening=False,  # apply ZCA whitening
                    rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
                    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
                    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
                    horizontal_flip=True,  # randomly flip images
                    vertical_flip=False)  # randomly flip images


# In[ ]:


if f is F.TRAIN:
    logging.info("Training baseline on full train set")
    
    datagen.fit(x_train)
    model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=epochs,
        validation_data=(x_test, y_test),
        steps_per_epoch=np.ceil(len(x_train)/128)
    )
    
    models["baseline"] = model
    model.save(os.path.join(output_dir, 'baseline.h5'))


# In[ ]:


calibration_metrics = {}


# In[ ]:


def evaluate_model(model, x_test, y_test, keras=False, bins=15):
    
    if keras:
        y_probs = model.predict(x_test, batch_size=128, verbose=0)
    else:
        y_probs = model.predict_proba(x_test)
    y_probs_max = y_probs.max(axis=1)
    y_preds = y_probs.argmax(axis=1)
    y_test = y_test.argmax(axis=1)
    
    return {
        "accuracy": sklearn.metrics.accuracy_score(y_test, y_preds),
        "balanced_accuracy": sklearn.metrics.balanced_accuracy_score(y_test, y_preds),
        'ece': metrics.ece(y_probs_max, y_preds, y_test, bins=bins),
        'ece_balanced': metrics.ece(y_probs_max, y_preds, y_test, balanced=True, bins=bins),
        'peace': metrics.peace(y_probs_max, y_preds, y_test, bins=bins),
        'class_wise_ece': metrics.class_wise_error(y_probs, y_preds, y_test, metrics.ece, bins=bins),
        'class_wise_ece_balanced': metrics.class_wise_error(y_probs, y_preds, y_test, metrics.ece, balanced=True, bins=bins),
        'class_wise_peace': metrics.class_wise_error(y_probs, y_preds, y_test, metrics.peace, bins=bins)
    }


# In[ ]:


calibration_metrics["baseline"] = evaluate_model(models["baseline"], x_test, y_test, keras=True)
del models["baseline"]


# In[ ]:


x_train_s, x_val, y_train_s, y_val = sklearn.model_selection.train_test_split(x_train, y_train.argmax(axis=1), test_size=0.25)

mean = x_train_s.mean()
std = x_train_s.std()

x_train_s = (x_train_s - mean) / (std + 1e-7)
y_train_s = tf.keras.utils.to_categorical(y_train_s)

x_val = (x_val - mean) / (std + 1e-7)
y_val = tf.keras.utils.to_categorical(y_val)


# In[ ]:


if f is F.TRAIN:
    logging.info("Training baseline on reduced train set")
    model.load_weights("models/model.h5")

    datagen.fit(x_train_s)
    model.fit(
        datagen.flow(x_train, y_train, batch_size=128),
        epochs=epochs,
        validation_data=(x_test, y_test),
        steps_per_epoch=np.ceil(len(x_train)/128)
    )

    models["baseline_s"] = model
    model.save(os.path.join(output_dir, 'baseline_s.h5'))


# In[ ]:


class CalibratedModel(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator=None, calibrator=None):
        ''' Initialize a Calibrated model (classifier + calibrator)

        Parameters
        ----------
        base_estimator : string
            Name of the classifier
        method : string
            Name of the calibrator
        '''
        self.base_estimator = base_estimator
        self.calibrator = calibrator

    def fit(self, X, y, X_val=None, y_val=None, *args, **kwargs):
        """Fit the calibrated model

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        y : array-like, shape (n_samples, n_classes)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        X, y = indexable(X, y)

        scores = self.base_estimator.predict(X, batch_size=128, verbose=1)

        if X_val is not None:
            X_val, y_val = indexable(X_val, y_val)
            # TODO add scores of validation
            scores_val = self.base_estimator.predict(X_val, batch_size=128, verbose=1)
        else:
            scores_val = None

        self.calibrator.fit(scores, y, X_val=scores_val, y_val=y_val, *args, **kwargs)
        
        return self

    def predict_proba(self, X):
        """Posterior probabilities of classification

        This function returns posterior probabilities of classification
        according to each class on an array of test vectors X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples, n_classes)
            The predicted probas. Can be exact zeros.
        """

        scores = self.base_estimator.predict(X, batch_size=256, verbose=1)

        predictions = self.calibrator.predict_proba(scores)

        return predictions

    def predict(self, X, *args):
        """Predict the target of new samples. Can be different from the
        prediction of the uncalibrated classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.

        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """
        check_is_fitted(self, ["calibrator"])
        return np.argmax(self.predict_proba(X), axis=1)
    


# In[ ]:


reg_lambda_list = [1, 0.1 , 0.01, 0.001, 0.0001]
reg_mu_list = [1, 0.1, 0.01, 0.001]


# In[ ]:


calibrators = {}


# In[ ]:


logging.info("Temperature scaling started")


# In[ ]:


calibrator = tempscaling.TemperatureScaling(reg_lambda_list=reg_lambda_list, reg_mu_list=reg_mu_list, logit_constant=0.0)
models["temperature"] = CalibratedModel(models["baseline_s"], calibrator)


# In[ ]:


models["temperature"].fit(x_train_s, y_train_s.argmax(axis=1), X_val=x_val, y_val=y_val.argmax(axis=1), verbose=0)


# In[ ]:


calibration_metrics["temperature"] = evaluate_model(models["temperature"], x_test, y_test)


# In[ ]:


logging.info("Temperature scaling ended")


# In[ ]:


logging.info("Dirichlet calibration started")


# In[ ]:


calibrator = DirichletCalibrator(matrix_type="full", l2=0.1)
models["dirichlet"] = CalibratedModel(models["baseline_s"], calibrator)
models["dirichlet"].fit(x_train_s, y_train_s.argmax(axis=1), X_val=x_val, y_val=y_val.argmax(axis=1), verbose=0)


# In[ ]:


calibration_metrics["dirichlet"] = evaluate_model(models["dirichlet"], x_test, y_test)


# In[ ]:


logging.info("Dirichlet calibration ended")


# In[ ]:


logging.info("Vector scaling started")


# In[ ]:


calibrator = vectorscaling.VectorScaling(reg_lambda_list=reg_lambda_list, reg_mu_list=reg_mu_list)
models["vector"] = CalibratedModel(models["baseline_s"], calibrator)
models["vector"].fit(x_train_s, y_train_s.argmax(axis=1), X_val=x_val, y_val=y_val.argmax(axis=1), verbose=0)


# In[ ]:


calibration_metrics["vector"] = evaluate_model(models["vector"], x_test, y_test)


# In[ ]:


logging.info("Vector scaling ended")


# In[ ]:


for k, model in models.items():
    if type(model) is tf.python.keras.engine.functional.Functional:
        fname = os.path.join(output_dir, f"{k}.h5")
        model.save(fname)
    else:
        fname = os.path.join(output_dir, f"{k}.h5")
        model.base_estimator = None
        dump(model, fname)


# In[ ]:


df = pd.DataFrame(calibration_metrics).T


# In[ ]:


dump(os.path.join(output_dir, "calibration_metrics.dat"), df)


# In[ ]:


if f is F.TRAIN:
    exit()


# In[ ]:


df[["ece", "peace", "class_wise_ece", "class_wise_peace"]].plot.bar().legend(loc='center left',bbox_to_anchor=(1.0, 0.15))


# In[ ]:


df.index[df["peace"].argmin()], df["peace"].min()


# In[ ]:


df.index[df["class_wise_peace"].argmin()], df["class_wise_peace"].min()


# In[ ]:


df.index[df["ece"].argmin()], df["ece"].min()


# In[ ]:


df.index[df["class_wise_ece"].argmin()], df["class_wise_ece"].min()


# In[ ]:


fig, axes = plt.subplots(1, len(models), figsize=(len(models)*3, 3), constrained_layout=True)
for ax, (k, model) in zip(axes, models.items()):
    
    ax.set_title(k)
    
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(x_test)
    else:
        y_probs = model.predict(x_test, batch_size=256, verbose=1)
    y_probs_max = y_probs.max(axis=1)
    y_preds = y_probs.argmax(axis=1)
    
    plots.posterior_reliability_diagram(y_probs_max, y_preds, y_test.argmax(axis=1), bins=15, ax=ax)


# In[ ]:


fig, axes = plt.subplots(2, 10, figsize=(25, 6), sharex=True, sharey=True, constrained_layout=True)
for row, (k, model) in zip(axes, {k:v for k,v in models.items() if k in ["temperature", "vector"]}.items()):

    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(x_test)
    else:
        y_probs = model.predict(x_test, batch_size=256, verbose=1)
    y_preds = y_probs.argmax(axis=1)

    plots.class_wise_posterior_reliability_diagram(y_probs, y_preds, y_test.argmax(axis=1), bins=10, axes=row, metric=metrics.peace)
    
    row[0].set_ylabel(k + "\n" + row[0].get_ylabel())


# In[ ]:




