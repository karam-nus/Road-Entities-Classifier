#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
NUS ISS
Pattern Recognition & Machine Learning Systems
Testing script for -
Continuous Assessment 2 : Road Object Image Classifier
Team : validation7407

@author: karam
'''

#supress future warnings due to version dependencies
import warnings
warnings.filterwarnings("ignore")
import logging
logging.getLogger('tensorflow').disabled = True


# 1. Import libraries-----------------------------------------



import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,Dropout
from tensorflow.keras.layers import Flatten,Dense,BatchNormalization,Activation
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras import optimizers,regularizers

from sklearn.metrics  import confusion_matrix
import sklearn.metrics as metrics



# 2. Initialize data---------------------------------------



test_dir = 'data/test'

modelname_150 = 'CA2_150_1_0509'
filepath_150 = modelname_150 + ".hdf5"
target_size_150 = (150,150)

modelname_300 = 'CA2_300_1_0509'
filepath_300 = modelname_300 + ".hdf5"
target_size_300 = (300,300)

test_datagen = ImageDataGenerator( rescale =1. / 255 )
test_generator_150 = test_datagen.flow_from_directory( test_dir,
                                                     batch_size  = 1,
                                                     class_mode  = 'categorical',
                                                     target_size = target_size_150,
                                                     shuffle=False)

test_generator_300 = test_datagen.flow_from_directory( test_dir,
                                                     batch_size  = 1,
                                                     class_mode  = 'categorical',
                                                     target_size = target_size_300,
                                                     shuffle=False)
# 3. Define models-----------------------------------------



def createModel_150():
    inputs = Input(shape=(150,150,3))
    x = Conv2D(32,(3,3),padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32,(3,3),padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(32,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32,(3,3),padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(64,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64,(3,3),padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(256,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(4, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='categorical_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])
    return model


def createModel_300():
    inputs = Input(shape=(300,300,3))
    x = Conv2D(32,(3,3),padding='same')(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(32,(3,3),padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(32,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(32,(3,3),padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(64,(3,3),padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(64,(3,3),padding='same')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(128,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.25)(x)
    x = Conv2D(128,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(256,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(2,2))(x)
    x = Conv2D(512,(3,3),padding='same',kernel_regularizer=regularizers.l2(0.01))(x)
    x = Activation('relu')(x)
    x = MaxPooling2D(pool_size=(4,4))(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.25)(x)
    x = Dense(4, activation='softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(loss='categorical_crossentropy',
                   optimizer='rmsprop',
                   metrics=['accuracy'])
    return model




model_150 = createModel_150()
model_150.load_weights(filepath_150)

model_300 = createModel_300()
model_300.load_weights(filepath_300)


# 4. Test data---------------------------------------------


print("Predicting labels with model trained on size 150: ")
y_predict_150 = model_150.predict_generator(test_generator_150,
                                      verbose=1)
predicted_lables_150 = np.argmax(y_predict_150, axis=1)
actual_labels_150 = test_generator_150.labels

print("Predicting labels with model trained on size 300: ")
y_predict_300 = model_300.predict_generator(test_generator_300,
                                      verbose=1)
predicted_lables_300 = np.argmax(y_predict_300, axis=1)
actual_labels_300 = test_generator_300.labels



# 5.Model accuracy-----------------------------------------

labelnames   = ['Bike','Bus','Car','Pedestrian']


testScores_150  = metrics.accuracy_score(actual_labels_150,predicted_lables_150)
confusion_150   = metrics.confusion_matrix(actual_labels_150,predicted_lables_150)

print('\n')
print("------------------------------------")
print("Best accuracy (on testing dataset with size 150): %.2f%%" % (testScores_150*100))
print("------------------------------------")
print(metrics.classification_report(actual_labels_150,predicted_lables_150,target_names=labelnames,digits=4))
print(confusion_150)

testScores_300  = metrics.accuracy_score(actual_labels_300,predicted_lables_300)
confusion_300   = metrics.confusion_matrix(actual_labels_300,predicted_lables_300)

print('\n')
print("------------------------------------")
print("Best accuracy (on testing dataset with size 300, " +' \n' +
      "trained through progressive resizing): %.2f%%" % (testScores_300*100))
print("------------------------------------")
print(metrics.classification_report(actual_labels_300,predicted_lables_300,target_names=labelnames,digits=4))
print(confusion_300)

print("Prediction complete.")