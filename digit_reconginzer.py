import numpy as np # for algebra
import pandas as pd # data processing
import plotly.express as px
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split


# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Print the shape of the Training Data and Test Data.

train.shape, test.shape

print('Train Data Sample : ')
train.head()

print('Test Data Sample : ')
test.head()

X_train1 = train.iloc[:,1:].values  # set training data
y_train = train.iloc[:,0].values # set training labels
X_train1 = X_train1.reshape(42000,28,28) # reshape into a format that can be fed into our NN

# Using Plotly we can visualize sample of train data for some examples
print(y_train[12]) # label value
px.imshow(X_train1[12],color_continuous_scale='hot') # Corresponding Data Value

print(y_train[25]) # label value
px.imshow(X_train1[25],color_continuous_scale='hot') # Corresponding Data Value

# Let's Print a Summary of our training data to get a description
train.describe()

# Normalization
# As all pixel values so the values will be ranging between 0-255 pixel units , data values can be normalized to have data scaled between 0 - 1

X_train1 = X_train1/255
X = X_train1
y = y_train

# Normalize the triainig data

X_train, X_valid, y_train, y_valid = train_test_split(X_train1, y_train, test_size=0.2, random_state=42) # split data into training and validation

# The same preprocessing and normalization on testing data

X_test  = test.iloc[:,0:].values
X_test = X_test / 255
X_test_org = X_test
X_test.shape

X_test = X_test.reshape(28000,28,28) # Respahing to a 2D image.
px.imshow(X_test[10],color_continuous_scale='hot') # Plotting example of Test data


# Requierd Libraries for modeling Convolutional Neural Network (CNN)

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras import layers
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.layers import BatchNormalization,Dense,Conv2D,Input,MaxPooling2D,Dropout,Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import plot_model

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2)

# Custom Activation Function
def leaky_relu(z, alpha=0.01):
    return tf.maximum(alpha*z, z)

def my_softplus(z): # return value is just tf.nn.softplus(z)
    return tf.math.log(tf.exp(z) + 1.0)

# Custom kernel initializer 
def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)
  
  
tf.random.set_seed(42)
np.random.seed(42)

input_ = Input((28,28,1))
z = Conv2D(32, kernel_size=5, strides = 1,activation='relu',kernel_regularizer=l2(0.00004),\
           kernel_initializer=my_glorot_initializer)(input_)
z = BatchNormalization()(z)
z = MaxPooling2D(strides = 2)(z)
z = Dropout(0.2)(z)

z = Conv2D(64, kernel_size=3, activation=leaky_relu,kernel_regularizer=l2(0.00004))(z)
z = Conv2D(64, kernel_size=3,use_bias=False, activation=leaky_relu)(z)
z = BatchNormalization()(z)
z = MaxPooling2D()(z)
z = Dropout(0.2)(z)
    
z = Flatten()(z)

z = Dense(256, use_bias=False,activation=leaky_relu)(z)
z = BatchNormalization()(z)

z = Dense(128, use_bias=False,activation=leaky_relu)(z)
z = BatchNormalization()(z)

z1 = Dense(64,activation=leaky_relu)(z)
z = BatchNormalization()(z1)

z2 = Dense(64,activation=leaky_relu)(z)
z = BatchNormalization()(z2)
z = Dropout(0.2)(z)
z = tf.keras.layers.Add()([z1,z2])
z = Dense(10, use_bias=False,activation=leaky_relu)(z)

outputs = Dense(units = 10, activation = my_softplus)(z)
model = Model(inputs=input_, outputs=outputs)

plot_model(model,show_shapes=True,show_layer_names=True)

model.summary()

def scheduler(ep, lr):
    
    if ep < 10:
        return lr
    else:
        return lr * tf.math.exp(-0.1)
    
callback = tf.keras.callbacks.LearningRateScheduler(scheduler) # set the callback to our scheduler function

model.compile(
    optimizer=Adam(), \
    loss = 'sparse_categorical_crossentropy', \
    metrics = ['accuracy'])

run = model.fit(X_train, y_train, epochs=50,callbacks=callback,validation_data=(X_valid, y_valid))

fig = px.line(y=model.history.history['accuracy'],title='Train Accuracy',template="plotly_dark")

fig.update_layout(
    title_font_color="#41BEE9", 
    xaxis=dict(color="#41BEE9",title='Epochs'), 
    yaxis=dict(color="#41BEE9",title='Accuracy') 
)
fig.show()

fig = px.line(y=model.history.history['val_accuracy'],title='Validation Accuracy',template="plotly_dark")

fig.update_layout(
    title_font_color="#41BEE9", 
    xaxis=dict(color="#41BEE9",title='Epochs'), 
    yaxis=dict(color="#41BEE9",title='Accuracy') 
)
fig.show()

fig = px.line(y=model.history.history['loss'],title='Train Loss',template="plotly_dark")

fig.update_layout(
    title_font_color="#41BEE9", 
    xaxis=dict(color="#41BEE9",title='Epochs'), 
    yaxis=dict(color="#41BEE9",title='Loss') 
)
fig.show()

fig = px.line(y=model.history.history['val_loss'],title='Validation Loss',template="plotly_dark")

fig.update_layout(
    title_font_color="#41BEE9", 
    xaxis=dict(color="#41BEE9",title='Epochs'), 
    yaxis=dict(color="#41BEE9",title='Loss') 
)
fig.show()

# Evaluation of the model

tval = model.evaluate(X_test,y_test)

X_test_org = np.array(X_test_org).reshape(-1, 28, 28, 1)
pred = model.predict(X_test_org)

predictions = np.argmax(pred,axis=1)

submissions = pd.read_csv('sample_submission.csv')

submissions['Label'] = predictions

submissions.head(10)

submissions.to_csv("submission.csv", index=False)

print('Train Accuracy : {}'.format(round(max(run.history['accuracy'])*100,4)))
print('Validation Accuracy : {}'.format(round(max(run.history['val_accuracy'])*100,4)))
print('Test Accuracy : {}'.format(round(tval[1]*100,4)))

# Thank You.
