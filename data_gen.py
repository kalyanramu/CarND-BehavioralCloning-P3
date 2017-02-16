import cv2
import sklearn
import os
import numpy as np
import random
import skimage.transform as sktransform

#0.2,0.125
def preprocess(image, top_offset=0.01, bottom_offset=0.01):
    """
    Applies preprocessing pipeline to an image: crops `top_offset` and `bottom_offset`
    portions of image, resizes to 32x128 px and scales pixel values to [0, 1].
    """
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    #image = cv2.resize(image[top:-bottom, :], (160, 320))
    image = sktransform.resize(image[top:-bottom, :], (160, 320, 3))
    #print(image.shape)
    return image


cameras = ["left", "center", "right"]


def generator(df_samples, datafolder_path, augument=False, batch_size=32):
    num_samples = len(df_samples)
    print("Num Samples:", num_samples)
    num_cameras = len(cameras)
    while 1:  # Loop forever so the generator never terminates
        # sklearn.utils.shuffle(df_samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = df_samples[offset:offset + batch_size]
            images = []
            angles = []
            # print(batch_samples) #Debug
            for index, batch_sample in batch_samples.iterrows():
                #name = datafolder_path+'/'+batch_sample['center']
                steer_angle = float(batch_sample['steering'])

                # Pick random camera : To teach how to steer from left to right, vice-versa
                # After adding this, the car was driving fine but keep ending
                # up in lake
                if augument:
                    rand_index = random.randint(1, num_cameras - 1)
                    random_camera = cameras[rand_index]
                    if random_camera == "left":
                        steer_angle += 0.25
                    elif random_camera == "right":
                        steer_angle -= 0.25
                else:
                    random_camera = "center"

                    # Get the image data
                    # print(datafolder_path)
                    # print(batch_sample[random_camera])
                    # Strip white spaces in pd frame
                name = os.path.join(datafolder_path, batch_sample[
                    random_camera].strip())
                img = cv2.imread(name)
                correct_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # print(name)
                # print(type(img))
                if type(img) == None:
                    print("Error reading file, check file path")
                    break
                
                if augument == True:
                    rand_val = random.random()
                    if rand_val < 0.5:
                        steer_angle *= -1.0
                        flip_img = cv2.flip(correct_img, 1)
                    else:
                        flip_img = correct_img
                else:
                    flip_img = correct_img

                final_img = flip_img

                # Accumulate the data
                images.append(final_img)
                angles.append(steer_angle)

            # trim image to only see section with road
            #X_train = X_train[:,80:,:,:]
            X_train = np.array(images)
            y_train = np.array(angles)
            #print(y_train)
            #print("X shape:",X_train.shape)
            # return sklearn.utils.shuffle(X_train, y_train) #for testing
            # purposes using test_data_gen.py
            yield sklearn.utils.shuffle(X_train, y_train)
            #return sklearn.utils.shuffle(X_train, y_train)
