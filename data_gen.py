import cv2
import sklearn
import os
import numpy as np
import random
import skimage.transform as sktransform

# 0.2,0.125


def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = 20 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(image1, cv2.COLOR_HSV2RGB)
    return image1


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
    # print(image.shape)
    crop_img = image[top_offset:height - bottom_offset]
    #gray_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
    #gray_img_exp = np.expand_dims(gray_img,axis=3)
    #final_img = gray_img_exp
    final_img = augment_brightness_camera_images(crop_img)
    return final_img


cameras = ["left", "center", "right"]


def generator(df_samples, datafolder_path, augument=False, batch_size=32, opname=""):
    print("Start of Epoch")
    num_samples = len(df_samples)
    print("Num Samples:", num_samples)
    num_cameras = len(cameras)
    print(opname)

    # Reset offset to zero at begin of epoch
    offset = 0

    # Create Histogram
    df = df_samples
    num_bins = 200  # N of bins
    bin_list = [None] * num_bins
    start = 0
    i = 0
    for end in np.linspace(0, 1, num=num_bins):
        df_range = np.array(df[(np.absolute(df.steering) >= start) & (
            np.absolute(df.steering) < end)].index)
        # print(end)
        bin_list[i] = df_range
        #print("Index ",i, " :")
        # print(df_range)
        start = end
        i += 1
    final_bin_list = bin_list
    bin_cnt = num_bins*[0]

    while 1:  # Loop forever so the generator never terminates
        # Get the Batch
        j = 0
        start_index = 0
        table_indices = []
        while j < batch_size:
            # get row index
            #start_index = np.random.randint(0,num_bins/4)
            start_index = 0
            row_index = np.random.randint(start_index, num_bins)
            num_vals_bin = len(final_bin_list[row_index])
            if num_vals_bin > 0:
                col_index = np.random.randint(num_vals_bin)
                table_index = final_bin_list[row_index][col_index]
                j += 1
                table_indices.append(table_index)
                bin_cnt[row_index] +=1
        batch_samples = df.loc[table_indices]
        #print(np.max(bin_cnt))
        images = []
        angles = []
        iter = 0
        for index, batch_sample in batch_samples.iterrows():
            #name = datafolder_path+'/'+batch_sample['center']
            # table_index = (index+offset)%num_samples #Cicular Buffer Index over the rows of the table
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

            # Read file from folder
            name = os.path.join(datafolder_path, batch_sample[
                                random_camera].strip())
            img = cv2.imread(name)
            correct_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if type(img) == None:
                print("Error reading file, check file path")
                break

            # Make alternate image steer pos and negative (Balance positive and
            # negative angles)
            if (iter % 2 == 0 and steer_angle < 0) or (iter % 2 != 0 and steer_angle > 0):
                flip_img = cv2.flip(correct_img, 1)
                steer_angle = steer_angle * -1.0 # invert the steering angle
            else:
                flip_img = correct_img

            # Crop the image
            #v_delta = .05 if augument else 0
            crop_img = preprocess(flip_img, top_offset=60, bottom_offset=20)
            final_img = crop_img
            #final_img = flip_img

            #steer_angle += 0.1*(np.random.rand()-0.5)
            # Accumulate the data
            images.append(final_img)
            angles.append(steer_angle)
            iter +=1
            if offset == 0 and iter == 0:
                cv2.imwrite("img"+str(offset),final_img)

        # trim image to only see section with road
        #X_train = X_train[:,80:,:,:]
        X_train = np.array(images)
        y_train = np.array(angles)

        # Append data to csv file
        if opname == "train":
            filename = "train_log.csv"
            f = open(filename, "ab")
            np.savetxt(f, y_train)
            f.close()

        # print(y_train)
        #print("X shape:",X_train.shape)
        # return sklearn.utils.shuffle(X_train, y_train) #for testing
        # purposes using test_data_gen.py
        yield sklearn.utils.shuffle(X_train, y_train)
        offset += batch_size

    # return sklearn.utils.shuffle(X_train, y_train)
