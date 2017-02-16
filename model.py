#Installs
#pip install h5py
#pip install keras
#pip install tensorflow-gpu
#pip install opencv-python

#Imports
import csv
import cv2
import numpy as np

#------------------------------------#
#Read the Spreadsheet
with open('./data/data/driving_log.csv') as csvfile:
	filereader = csv.reader(csvfile, delimiter=',')
	lines=[]
	for row in filereader:
		lines.append(row)

#------------------------------------#
#Collect Image Info and Steering Angles
#print(lines[23])
images=[]
steer_angles=[]

for line in lines[1:]:
	src = "./data/data/"+line[0]	#first column	
	steer_angles.append(float(line[-1])) 	#last column
	#print(src)
	img = cv2.imread(src)
	final_img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	images.append(final_img)
X_train = np.array(images)
y_train = np.array(steer_angles)



#Print Summary
print("Number of Training Set:", len(images))
print("Number of Angles:",len(steer_angles))
#print("Images Shape: ",X_train.shape)

#-------------------------------------#
#Create Model in keras and train
import keras
from keras.models import Sequential
from keras.layers.core import  Flatten,Dense,Dropout
from keras.layers.core import Lambda
from keras.layers.convolutional import  Convolution2D
from keras.layers.pooling import MaxPooling2D


model = Sequential()
model.add(Lambda(lambda x: x/255.0 -0.5,input_shape=(160, 320, 3)))
model.add(Convolution2D(16,3,3,activation='elu', name='block1_conv'))
model.add(MaxPooling2D(pool_size=(2,2),name='block1_pool'))
model.add(Convolution2D(32,3,3,activation='elu',name='block1_relu'))

model.add(MaxPooling2D(pool_size=(2,2),name='block2_conv'))
model.add(Convolution2D(64,3,3,activation='relu',name='block2_relu'))
model.add(MaxPooling2D(pool_size=(2,2),name='block2_pool'))

model.add(Flatten())
model.add(Dense(500,activation='relu',name='block3_fc'))
model.add(Dropout(0.5))
model.add(Dense(100,activation='relu',name='block4_fc'))
model.add(Dropout(0.25))
model.add(Dense(20,activation='relu',name='block5_fc'))
model.add(Dense(1))

model.compile(optimizer='adam',loss='mse')
model.fit(X_train,y_train,validation_split=0.25,batch_size=64,nb_epoch=20,shuffle=True)


model.save('model.h5')
