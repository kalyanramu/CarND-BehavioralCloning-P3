import cv2
import sklearn
import os
import numpy as np
import random
import skimage.transform as sktransform

#0.2,0.125
def preprocess(image, top_offset=20, bottom_offset=20):
    """
    Applies preprocessing pipeline to an image: crops `top_offset` and `bottom_offset`
    portions of image, resizes to 32x128 px and scales pixel values to [0, 1].
    """
    input_size = image.shape
    height = input_size[0]
    #top = int(top_offset * image.shape[0])
    #bottom = int(bottom_offset * image.shape[0])
    #image = cv2.resize(image[top:-bottom, :], (160, 320))
    #image = sktransform.resize(image[top:-bottom, :], (160, 320, 3))
    #image = sktransform.resize(image[top:height-bottom, :], input_size)
    #print(image.shape)
    crop_img = image[top_offset:height-bottom_offset]
    #gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    #gray_img_exp = np.expand_dims(gray_img,axis=3)
    #final_img = gray_img_exp
    final_img = crop_img
    return final_img


cameras = ["left", "center", "right"]


def generator(df_samples, datafolder_path, augument=False, batch_size=32,name=""):
    print("Start of Epoch")
    num_samples = len(df_samples)
    print("Num Samples:", num_samples)
    num_cameras = len(cameras)
    print(name)
    #----------------Remove Zero Steer Data---------------#
    zero_idx=df_samples[abs(df_samples['steering'])< 0.01].index
    print(len(zero_idx))
    #Get subset/Filter zero steer data
    pkeep =0.0 #Keep only X% of samples
    drop_idx = np.random.choice(zero_idx,int(len(zero_idx)*(1-pkeep)))
    df = df_samples.drop(drop_idx)
    print("Trimmed Samples",len(df))

    #Reset offset to zero at begin of epoch
    offset =0
    filename = "train_log.csv"
    f = open(filename, "w+")
    f.close()
    while 1:  # Loop forever so the generator never terminates
        seq_indices = np.array(range(offset,offset+batch_size)) #get seq indices
        #print("Seq Indices:", seq_indices)
        indices = seq_indices %(len(df)-1) #As we are doing aug, get circular buffer indices

        batch_samples = df.iloc[indices] #Don't use .ix, it is buggy
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
            crop_img = preprocess(flip_img,top_offset=60,bottom_offset=20)

            #Convert to Greyscale


            final_img = crop_img
            #final_img = flip_img


            # Accumulate the data
            images.append(final_img)
            angles.append(steer_angle)
                
        
        # trim image to only see section with road
        #X_train = X_train[:,80:,:,:]
        X_train = np.array(images)
        y_train = np.array(angles)

        #Append data to csv file
        f=open('foo.csv','ab')
        np.savetxt(f,y_train)
        f.close()
    
        #print(y_train)
        #print("X shape:",X_train.shape)
        # return sklearn.utils.shuffle(X_train, y_train) #for testing
        # purposes using test_data_gen.py
        yield sklearn.utils.shuffle(X_train, y_train)
        offset += batch_size
    #return sklearn.utils.shuffle(X_train, y_train)
