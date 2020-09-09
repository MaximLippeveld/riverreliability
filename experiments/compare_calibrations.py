#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from ridgereliability import metrics, plots
import sklearn.model_selection
import sklearn.metrics


# In[2]:


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


# In[3]:


dev = [d for d in tf.config.experimental.list_logical_devices() if d.device_type=="GPU"][0]


# In[4]:


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

mean = x_train.mean()
std = x_train.std()

x_test = (x_test - mean) / (std + 1e-7)
y_test = tf.keras.utils.to_categorical(y_test)


# In[5]:


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

model.save_weights('models/model.h5')


# In[6]:


x_train = (x_train - mean) / (std + 1e-7)
y_train = tf.keras.utils.to_categorical(y_train)

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
datagen.fit(x_train)


# In[ ]:


model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=1,
    validation_data=(x_test, y_test),
    steps_per_epoch=np.ceil(len(x_train)/128)
)

model.save('models/cifar10_densenet_baseline.h5')


# In[9]:


x_train_s, x_val, y_train_s, y_val = sklearn.model_selection.train_test_split(x_train, y_train, test_size=0.25)

mean = x_train_s.mean()
std = x_train_s.std()

x_train_s = (x_train_s - mean) / (std + 1e-7)
y_train_s = tf.keras.utils.to_categorical(y_train_s)

x_val = (x_val - mean) / (std + 1e-7)
y_val = tf.keras.utils.to_categorical(y_val)


# In[10]:


with tf.device(dev.name):
    datagen_s = tf.keras.preprocessing.image.ImageDataGenerator(
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
datagen_s.fit(x_train_s)


# In[11]:


model.load_weights("models/model.h5")


# In[12]:


model.fit(
    datagen.flow(x_train, y_train, batch_size=128),
    epochs=1,
    validation_data=(x_test, y_test),
    steps_per_epoch=np.ceil(len(x_train)/128)
)

model.save('models/cifar10_densenet_baseline_s.h5')


# In[18]:


exit()


# In[24]:


from importlib import reload
reload(plots)


# In[26]:


def evaluate_model(model, x_test, y_test):
    
    y_probs = model.predict(x_test, batch_size=128, verbose=1)
    y_probs_max = y_probs.max(axis=1)
    y_preds = y_probs.argmax(axis=1)
    y_test = y_test.argmax(axis=1)
    
    plots.posterior_reliability_diagram(y_probs_max, y_preds, y_test, bins=15)
    
    return {
        "accuracy": sklearn.metrics.accuracy_score(y_test, y_preds),
        "balanced_accuracy": sklearn.metrics.balanced_accuracy_score(y_test, y_preds),
        'ece': metrics.ece(y_probs_max, y_preds, y_test),
        'ece_balanced': metrics.ece(y_probs_max, y_preds, y_test, balanced=True),
        'peace': metrics.peace(y_probs_max, y_preds, y_test),
        'class_wise_ece': metrics.class_wise_error(y_probs, y_preds, y_test, metrics.ece),
        'class_wise_ece_balanced': metrics.class_wise_error(y_probs, y_preds, y_test, metrics.ece, balanced=True),
        'class_wise_peace': metrics.class_wise_error(y_probs, y_preds, y_test, metrics.peace)
    }


# In[29]:


evaluate_model(baseline_model, x_test, y_test)


# In[30]:


evaluate_model(baseline_model_s, x_test, y_test)


# In[ ]:




