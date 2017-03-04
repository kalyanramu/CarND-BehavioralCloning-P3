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

act_sel1 = 'relu'
act_sel2 = 'relu'
model = Sequential()
#model.add(Lambda(lambda x: x/255.0 -0.5,input_shape=(160, 320, 3)))
model.add(Lambda(lambda x: x/255.0 -0.5,input_shape=(70, 320, 3))) #Crop Image
#model.add(Lambda(lambda x: x/255.0 -0.5,input_shape=(80, 320, 1))) #Crop and Gray
model.add(Convolution2D(16,3,3, activation=act_sel1, name='block1_conv'))
model.add(MaxPooling2D(pool_size=(2,2),name='block1_pool'))
#model.add(Dropout(0.9))

model.add(Convolution2D(32,3,3, activation=act_sel1,name='block1_relu'))
model.add(MaxPooling2D(pool_size=(2,2),name='block2_conv'))
#model.add(Dropout(0.9))

model.add(Convolution2D(64,3,3, activation=act_sel1,name='block2_relu'))
model.add(MaxPooling2D(pool_size=(2,2),name='block2_pool'))
#model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(500,activation=act_sel2,name='block3_fc'))
#model.add(Dropout(0.5))

model.add(Dense(200,activation=act_sel2,name='block4_fc'))
#model.add(Dropout(0.5))

#model.add(Dense(100,activation=act_sel2,name='block5_fc'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse',lr=1e-4) #Setting this learning rate slow was crucial, 1e-3 was too high
#model.fit(X_train,y_train,validation_split=0.25,batch_size=64,nb_epoch=20,shuffle=True)
batch_size = 32

#Log data in training
filename = "train_log.csv"
f=open(filename,'wb')
np.savetxt(f,np.array([]))
f.close()

print("Samples:",len(df))
history = model.fit_generator(generator(df,datafolder_path="./data/data",augument=True,opname="train"),samples_per_epoch = 500*batch_size,nb_epoch=10,validation_data=generator(df_valid,datafolder_path="./data/data",augument=True,opname="valid"),nb_val_samples=4*batch_size)

model.save('model.h5')

from keras import backend
backend.clear_session() #Clear memory of tensorflow --- magic wand