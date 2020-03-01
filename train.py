#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
NUS ISS
Pattern Recognition & Machine Learning Systems
Training script for -
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




# 2. Initialize values-------------------------------------



modelname_150 = 'CA2_150_1_0509'

train_dir = 'data/train'
val_dir = 'data/val'

target_size_150 = (150, 150)
target_size_300 = (300, 300)

batch_size = 16

seed = 1
np.random.seed(seed)



# 3. Data Augmentation-------------------------------------



train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2,
                                   zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)

print("Training Data")
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=target_size_150,
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    seed=1)
print("Validation Data")
validation_generator = val_datagen.flow_from_directory(val_dir,
                                                        target_size=target_size_150,
                                                        color_mode='rgb',
                                                        batch_size=batch_size,
                                                        class_mode='categorical',
                                                        seed=1)



# 4. Model Definition-------------------------------------



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



# 5. Model Training-------------------------------------



model_150 = createModel_150()
print("Model summary :::"+'\n')
print(modelname_150)
model_150.summary()

#Model diagram
plot_model(model_150,
             to_file=modelname_150+'.png',
             show_shapes=True,
             show_layer_names=False,
             rankdir='TB')



#Defining checkpoint
filepath_150 = modelname_150 + ".hdf5"
checkpoint = ModelCheckpoint(filepath_150,
                             monitor='val_acc',
                             verbose=0,
                             save_best_only=True,
                             mode='max')
csv_logger = CSVLogger(modelname_150 + '.csv')
callbacks_list = [checkpoint, csv_logger]


start = time.time()
model_150.fit_generator(
     train_generator,
     steps_per_epoch=2000 // batch_size,
     epochs=120,
     validation_data=validation_generator,
     validation_steps=800 // batch_size,
     callbacks=callbacks_list)
end = time.time()

print('Training duration for model ::: %.2f seconds' % (end - start))



# 6. Training result plot----------------------------------



print("Training Result :::")
records = pd.read_csv(modelname_150 + '.csv')
print(records['val_acc'].max())

plt.figure(figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
plt.style.use('ggplot')
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'


plt.subplot(211)
plt.plot(records['val_loss'])
plt.plot(records['loss'])
plt.title('Loss value',fontsize=12)

plt.subplot(212)
plt.plot(records['val_acc'])
plt.plot(records['acc'])
plt.title('Accuracy',fontsize=12)
plt.show()



# 7. Model tuning through progressive resizing----------------


print("Training a new model (input size = 300*300) with weights of previous model " + '\n'+
								" (having input size = 150*150) for better accuracy.")
modelname_300 = 'CA2_300_1_0509'
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)

print("Training Data:")
train_generator = train_datagen.flow_from_directory(train_dir,
                                                    target_size=target_size_300,
                                                    color_mode='rgb',
                                                    batch_size=batch_size,
                                                    class_mode='categorical')
print("Validation Data:")
validation_generator = val_datagen.flow_from_directory(val_dir,
                                                        target_size=target_size_300,
                                                        color_mode='rgb',
                                                        batch_size=batch_size,
                                                        class_mode='categorical')


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


model_300 = createModel_300()
model_300.load_weights(filepath_150)
print(modelname_300)
print("Model summary :::"+'\n')
model_300.summary()

plot_model(model_300,
             to_file=modelname_300+'.png',
             show_shapes=True,
             show_layer_names=False,
             rankdir='TB')

filepath_300 = modelname_300 + ".hdf5"
checkpoint = ModelCheckpoint(filepath_300,
                             monitor='val_acc',
                             verbose=0,
                             save_best_only=True,
                             mode='max')
csv_logger = CSVLogger(modelname_300 + '.csv')
callbacks_list = [checkpoint, csv_logger]

start = time.time()
model_300.fit_generator(
     train_generator,
     steps_per_epoch=2000 // batch_size,
     epochs=100,
     validation_data=validation_generator,
     validation_steps=800 // batch_size,
     callbacks=callbacks_list)
end = time.time()

print('Training duration for new model::: %.2f seconds' % (end - start))



# 6. Training result plot----------------------------------



print("Training Result :::")
records = pd.read_csv(modelname_300 +'.csv')
print(records['val_acc'].max())

plt.figure(figsize=(15, 5), dpi=80, facecolor='w', edgecolor='k')
plt.style.use('ggplot')
plt.rcParams['ytick.right']     = True
plt.rcParams['ytick.labelright']= True
plt.rcParams['ytick.left']      = False
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['font.family']     = 'Arial'


plt.subplot(211)
plt.plot(records['val_loss'])
plt.plot(records['loss'])
plt.title('Loss value',fontsize=12)

plt.subplot(212)
plt.plot(records['val_acc'])
plt.plot(records['acc'])
plt.title('Accuracy',fontsize=12)
plt.show()

print("Training Complete.")