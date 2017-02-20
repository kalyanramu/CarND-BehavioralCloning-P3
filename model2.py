#Installs
#pip install h5py
#pip install keras
#pip install tensorflow-gpu
#pip install opencv-python

#Imports
import csv
import cv2
import numpy as np
import pandas as pd
from data_gen import generator
from keras.regularizers import l2

#------------------------------------#
#Read data from file
from sklearn.model_selection import train_test_split
df = pd.read_csv('./data/data/driving_log.csv')
#df = pd.read_csv('./data/data/driving_log_balanced.csv')
df_train,df_valid = train_test_split(df, test_size=0.2)

#-------------------------------------#
#Create Model in keras and train
import keras
from keras.models import Sequential
from keras.layers.core import  Flatten,Dense,Dropout
from keras.layers.core import Lambda
from keras.layers.convolutional import  Convolution2D
from keras.layers.pooling import MaxPooling2D

reg_val = 0.01
INIT = 'glorot_uniform'

model = Sequential()
#model.add(Lambda(lambda x: x/255.0 -0.5,input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x/255.0 -0.5,input_shape=(80, 320, 3)))
model.add(Convolution2D(16,3,3,activation='relu', name='block1_conv'))
model.add(MaxPooling2D(pool_size=(2,2),name='block1_pool'))
model.add(Convolution2D(32,3,3,activation='relu',name='block1_relu'))

model.add(MaxPooling2D(pool_size=(2,2),name='block2_conv'))
model.add(Convolution2D(64,3,3,activation='relu',name='block2_relu'))
model.add(MaxPooling2D(pool_size=(2,2),name='block2_pool'))

model.add(Flatten())
model.add(Dense(500,activation='relu',name='block3_fc'))
model.add(Dropout(0.1))
model.add(Dense(100,activation='relu',name='block4_fc'))
model.add(Dropout(0.1))
model.add(Dense(20,activation='relu',name='block5_fc'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
#model.fit(X_train,y_train,validation_split=0.25,batch_size=64,nb_epoch=20,shuffle=True)
batch_size = 32

#Log data in training
filename = "train_log.csv"
# opening the file with w+ mode truncates the file
f = open(filename, "w+")
f.close()

print("Samples:",len(df))
history = model.fit_generator(generator(df,datafolder_path="./data/data",augument=True,name="train"),samples_per_epoch = 100*batch_size,nb_epoch=10,validation_data=generator(df_valid,datafolder_path="./data/data",augument=True,name="valid"),nb_val_samples=4*batch_size)

model.save('model.h5')

from keras import backend
backend.clear_session() #Clear memory of tensorflow --- magic wand