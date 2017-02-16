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
from sklearn.model_selection import train_test_split
import random

steering_angle =0.10001
throttle = 0.2345
print("steer= {:2f}, throttle={:2f}".format(steering_angle,throttle))

#------------------------------------#
#Read the Spreadsheet
df = pd.read_csv('./data/data/driving_log.csv')
df_train,df_test = train_test_split(df, test_size=5)
print(df_test)


test_img=cv2.imread('./data/data/IMG/center_2016_12_01_13_30_48_287.jpg')
print(type(test_img))
print(test_img.shape)

#from data_gen import generator
#generator(df,datafolder_path="./data/data")
#print(df.sample(frac=0.05))
