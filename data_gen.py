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
    input_size = image.shape
    top = int(top_offset * image.shape[0])
    bottom = int(bottom_offset * image.shape[0])
    #image = cv2.resize(image[top:-bottom, :], (160, 320))
    #image = sktransform.resize(image[top:-bottom, :], (160, 320, 3))
    image = sktransform.resize(image[top:-bottom, :], input_size)
    #print(image.shape)
    return image


cameras = ["left", "center", "right"]


def generator(df_samples, datafolder_path, augument=False, batch_size=32):
    num_samples = len(df_samples)
    print("Num Samples:", num_samples)
    num_cameras = len(cameras)

    print("Start of Epoch")
    #----------------Remove Zero Steer Data---------------#
    #Indices where steer!=0
    non_zero_idx= df_samples[df_samples['steering']!=0].index
    #indices where steer ==0
    zero_idx=df_samples[df_samples['steering']==0].index
    #Get subset of zero steer data
    sel_zero_idx = np.random.choice(zero_idx,int(len(zero_idx)*0.1))
    #Append non-zero steer and zero-steer data
    sel_indices= np.concatenate([non_zero_idx,sel_zero_idx])
    print("Max Index:",np.max(sel_indices))
    #print(non_zero_idx[0:3])

    #Reset offset to zero at begin of epoch
    offset =0
    print("Offset at begin of Epoch: ",offset)
    while 1:  # Loop forever so the generator never terminates
        # sklearn.utils.shuffle(df_samples)
        for i in range(1): #Made it one so that code need not be indented from previous version
            #batch_samples = df_samples[offset:offset + batch_size]
            #indices = np.array(range(offset,offset + batch_size))%(num_samples-1) #Circular Buffer Index over the rows of the table
            #indices = sel_indices[offset:offset+batch_size]
            print("Offset:",offset)
            seq_indices = np.array(range(offset,offset+batch_size)) #get seq indices
            #print("Seq Indices:", seq_indices)
            circular_indices = seq_indices %(len(sel_indices)-1) #As we are doing aug, get circular buffer indices
            indices = sel_indices[circular_indices] #Get indices for reduced zero-steer data
            #print("Final Indices:", indices)
            print(indices)
            batch_samples = df_samples.iloc[indices] #Don't use .ix, it is buggy

            #print(indices)
            images = []
            angles = []

            for index,batch_sample in batch_samples.iterrows():
                #name = datafolder_path+'/'+batch_sample['center']
                #table_index = (index+offset)%num_samples #Cicular Buffer Index over the rows of the table
                #batch_sample = df_samples[i]
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

                #Read file from folder
                name = os.path.join(datafolder_path, batch_sample[
                    random_camera].strip())
                img = cv2.imread(name)
                correct_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                if type(img) == None:
                    print("Error reading file, check file path")
                    break
                
                #Flip image
                if augument == True:
                    rand_val = random.random()
                    if rand_val < 0.5:
                        steer_angle *= -1.0
                        flip_img = cv2.flip(correct_img, 1)
                    else:
                        flip_img = correct_img
                else:
                    flip_img = correct_img

                #Crop the image
                #v_delta = .05 if augument else 0
                #crop_img = preprocess(flip_img,top_offset=random.uniform(.25 - v_delta, .25 + v_delta),bottom_offset=random.uniform(.125 - v_delta, .125 + v_delta))
                #final_img = crop_img
                final_img = flip_img

                # Accumulate the data
                images.append(final_img)
                angles.append(steer_angle)
                
            offset += batch_size
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
