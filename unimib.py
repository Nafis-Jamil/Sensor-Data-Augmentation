import os
import shutil #https://docs.python.org/3/library/shutil.html
from shutil import unpack_archive # to unzip
#from shutil import make_archive # to create zip for storage
import requests #for downloading zip file
from scipy import io #for loadmat, matlab conversion
import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt # for plotting - pandas uses matplotlib
from tabulate import tabulate # for verbose tables
from tensorflow.keras.utils import to_categorical # for one-hot encoding

import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
from scipy import stats
from pylab import rcParams
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras import optimizers
from keras.callbacks import EarlyStopping
from keras.layers import BatchNormalization
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import Dropout
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras import optimizers
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten,Activation
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Bidirectional
from keras.layers import TimeDistributed
from keras.layers import Input
from keras import Model
from keras.layers import Concatenate, BatchNormalization
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from keras.utils import to_categorical
from matplotlib import pyplot

import pickle

import matplotlib
#matplotlib.use('Agg')   # Use Agg backend to save figures
import matplotlib.pyplot as plt

import os
import shutil
import urllib.request # to get files from web w/o !wget
from scipy import io #for loadmat, matlab conversion
import numpy as np
import matplotlib.pyplot as plt # for plotting training curves

# to measure and display training time
import time
from datetime import timedelta

# model library and functions
from tensorflow import keras #added to save model
from tensorflow.keras import layers #format matches MNIST example
from tensorflow.keras.callbacks import EarlyStopping

# for computing and displaying output metrics
import seaborn as sns
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

# temp - needed for SHL split
import numpy as np
import matplotlib.pyplot as plt



from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation
from dtaidistance import dtw



from sklearn.model_selection import train_test_split


#%%
def DA_Jitter(X, sigma=0.03):
    myNoise = np.random.normal(loc=0, scale=sigma, size=X.shape)
    return X+myNoise

def left_shift(X, r):
    a=X[r:,:]
    b=X[:r,:]
    shifted= np.concatenate((a,b),axis=0)
    return shifted

def right_shift(X, r):
    a=X[-r:,:]
    b=X[:-r,:]
    shifted= np.concatenate((a,b),axis=0)
    return shifted

def DA_Scaling(X, sigma=0.1):
    scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1,X.shape[1])) # shape=(1,3)
    myNoise = np.matmul(np.ones((X.shape[0],1)), scalingFactor)
    return X*myNoise

nPerm = 4
minSegLength = 100
def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return(X_new)

from scipy.interpolate import CubicSpline

sigma = 0.2
knot = 4
## This example using cubic splice is not the best approach to generate random curves.
## You can use other aprroaches, e.g., Gaussian process regression, Bezier curve, etc.
def GenerateRandomCurves(X, sigma=0.2, knot=4):
    xx = (np.ones((X.shape[1],1))*(np.arange(0,X.shape[0], (X.shape[0]-1)/(knot+1)))).transpose()
    yy = np.random.normal(loc=1.0, scale=sigma, size=(knot+2, X.shape[1]))
    x_range = np.arange(X.shape[0])
    cs_list = []
    for i in range(X.shape[1]):
      cs_list.append(CubicSpline(xx[:, i], yy[:, i])(x_range))
    # cs_x = CubicSpline(xx[:,0], yy[:,0])
    # cs_y = CubicSpline(xx[:,1], yy[:,1])
    # cs_z = CubicSpline(xx[:,2], yy[:,2])
    # return np.array([cs_x(x_range),cs_y(x_range),cs_z(x_range)]).transpose()
    return np.array(cs_list).transpose()

def DA_MagWarp(X, sigma = 0.2):
    return X * GenerateRandomCurves(X, sigma)



sigma = 0.2
knot = 4

def DistortTimesteps(X, sigma=0.2):
    tt = GenerateRandomCurves(X, sigma) # Regard these samples aroun 1 as time intervals
    tt_cum = np.cumsum(tt, axis=0)        # Add intervals to make a cumulative graph
    # Make the last value to have X.shape[0]
    t_scale = [(X.shape[0]-1)/tt_cum[-1,0],(X.shape[0]-1)/tt_cum[-1,1],(X.shape[0]-1)/tt_cum[-1,2]]
    tt_cum[:,0] = tt_cum[:,0]*t_scale[0]
    tt_cum[:,1] = tt_cum[:,1]*t_scale[1]
    tt_cum[:,2] = tt_cum[:,2]*t_scale[2]
    return tt_cum

def DA_TimeWarp(X, sigma=0.2):
    tt_new = DistortTimesteps(X, sigma)
    X_new = np.zeros(X.shape)
    x_range = np.arange(X.shape[0])
    X_new[:,0] = np.interp(x_range, tt_new[:,0], X[:,0])
    X_new[:,1] = np.interp(x_range, tt_new[:,1], X[:,1])
    X_new[:,2] = np.interp(x_range, tt_new[:,2], X[:,2])
    return X_new


def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_new = np.zeros(X.shape)
    idx = np.random.permutation(nPerm)
    bWhile = True
    while bWhile == True:
        segs = np.zeros(nPerm+1, dtype=int)
        segs[1:-1] = np.sort(np.random.randint(minSegLength, X.shape[0]-minSegLength, nPerm-1))
        segs[-1] = X.shape[0]
        if np.min(segs[1:]-segs[0:-1]) > minSegLength:
            bWhile = False
    pp = 0
    for ii in range(nPerm):
        x_temp = X[segs[idx[ii]]:segs[idx[ii]+1],:]
        X_new[pp:pp+len(x_temp),:] = x_temp
        pp += len(x_temp)
    return(X_new)


def RandSampleTimesteps(X, nSample=1000):
    X_new = np.zeros(X.shape)
    tt = np.zeros((nSample,X.shape[1]), dtype=int)
    tt[1:-1,0] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[1:-1,1] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[1:-1,2] = np.sort(np.random.randint(1,X.shape[0]-1,nSample-2))
    tt[-1,:] = X.shape[0]-1
    return tt





def DA_RandSampling(X, nSample=1000):
    tt = RandSampleTimesteps(X, nSample)
    X_new = np.zeros(X.shape)
    X_new[:,0] = np.interp(np.arange(X.shape[0]), tt[:,0], X[tt[:,0],0])
    X_new[:,1] = np.interp(np.arange(X.shape[0]), tt[:,1], X[tt[:,1],1])
    X_new[:,2] = np.interp(np.arange(X.shape[0]), tt[:,2], X[tt[:,2],2])
    return X_new

def DA_Rotation(X):
    axis = np.random.uniform(low=-1, high=1, size=X.shape[1])
    angle = np.random.uniform(low=-np.pi, high=np.pi)
    return np.matmul(X , axangle2mat(axis,angle))




import numpy as np

def random_mask_data(data, mask_ratio=0.2):
    """
    Randomly mask a portion of the accelerometer sensor data.

    Parameters:
    - data: Numpy array of shape (num_samples, num_features) representing sensor data.
    - mask_ratio: The ratio of data to mask (between 0 and 1).

    Returns:
    - masked_data: Numpy array with masked data.
    """
    num_samples, num_features = data.shape
    mask_indices = np.random.choice(num_samples, int(num_samples * mask_ratio), replace=False)
    
    masked_data = data.copy()
    masked_data[mask_indices] = 0.0  # Set masked data points to zero
    
    return masked_data


import numpy as np

# Random Masking
def random_mask_data(data, mask_ratio=0.2):
    num_samples, num_features = data.shape
    mask_indices = np.random.choice(num_samples, int(num_samples * mask_ratio), replace=False)
    
    masked_data = data.copy()
    masked_data[mask_indices] = 0.0  # Set masked data points to zero
    
    return masked_data

# Zero Masking
def zero_mask_data(data, mask_ratio=0.2):
    num_samples, num_features = data.shape
    num_to_mask = int(num_samples * mask_ratio)
    
    masked_data = data.copy()
    masked_indices = np.random.choice(num_samples, num_to_mask, replace=False)
    
    for idx in masked_indices:
        masked_data[idx, :] = 0.0  # Set the entire row to zero
    
    return masked_data

# Gaussian Noise Masking
def gaussian_noise_mask_data(data, noise_stddev=0.1, mask_ratio=0.2):
    num_samples, num_features = data.shape
    num_to_mask = int(num_samples * mask_ratio)
    
    masked_data = data.copy()
    masked_indices = np.random.choice(num_samples, num_to_mask, replace=False)
    
    for idx in masked_indices:
        noise = np.random.normal(0, noise_stddev, num_features)
        masked_data[idx, :] += noise
    
    return masked_data

# Time-Based Masking
def time_based_mask_data(data, mask_periods):
    num_samples, num_features = data.shape
    
    masked_data = data.copy()
    
    for start, end in mask_periods:
        start_idx = int(start * num_samples)
        end_idx = int(end * num_samples)
        masked_data[start_idx:end_idx, :] = 0.0  # Set data in the specified intervals to zero
    
    return masked_data



#%%

#credit https://stackoverflow.com/questions/9419162/download-returned-zip-file-from-url
#many other methods I tried failed to download the file properly
def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def unimib_load_dataset(
    verbose = True,
    incl_xyz_accel = True, #include component accel_x/y/z in ____X data
    incl_rms_accel = False, #add rms value (total accel) of accel_x/y/z in ____X data
    incl_val_group = False, #True => returns x/y_test, x/y_validation, x/y_train
                           #False => combine test & validation groups
    split_subj = dict
                (train_subj = [4,5,6,7,8,10,11,12,14,15,19,20,21,22,24,26,27,29],
                validation_subj = [1,9,16,23,25,28],
                test_subj = [2,3,13,17,18,30]),
    one_hot_encode = True):
    #Download and unzip original dataset
    if (not os.path.isfile('./UniMiB-SHAR.zip')):
        print("Downloading UniMiB-SHAR.zip file")
        #invoking the shell command fails when exported to .py file
        #redirect link https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip
        #!wget https://www.dropbox.com/s/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip
        download_url('https://www.dropbox.com/s/raw/x2fpfqj0bpf8ep6/UniMiB-SHAR.zip','./UniMiB-SHAR.zip')
    if (not os.path.isdir('./UniMiB-SHAR')):
        shutil.unpack_archive('./UniMiB-SHAR.zip','.','zip')
    #Convert .mat files to numpy ndarrays
    path_in = './UniMiB-SHAR/data'
    #loadmat loads matlab files as dictionary, keys: header, version, globals, data
    adl_data = io.loadmat(path_in + '/adl_data.mat')['adl_data']
    adl_names = io.loadmat(path_in + '/adl_names.mat', chars_as_strings=True)['adl_names']
    adl_labels = io.loadmat(path_in + '/adl_labels.mat')['adl_labels']

    if(verbose):
        headers = ("Raw data","shape", "object type", "data type")
        mydata = [("adl_data:", adl_data.shape, type(adl_data), adl_data.dtype),
                ("adl_labels:", adl_labels.shape ,type(adl_labels), adl_labels.dtype),
                ("adl_names:", adl_names.shape, type(adl_names), adl_names.dtype)]
        print(tabulate(mydata, headers=headers))
    #Reshape data and compute total (rms) acceleration
    num_samples = 151 
    #UniMiB SHAR has fixed size of 453 which is 151 accelX, 151 accely, 151 accelz
    adl_data = np.reshape(adl_data,(-1,num_samples,3), order='F') #uses Fortran order
    if (incl_rms_accel):
        rms_accel = np.sqrt((adl_data[:,:,0]**2) + (adl_data[:,:,1]**2) + (adl_data[:,:,2]**2))
        adl_data = np.dstack((adl_data,rms_accel))
    #remove component accel if needed
    if (not incl_xyz_accel):
        adl_data = np.delete(adl_data, [0,1,2], 2)
    if(verbose):
        headers = ("Reshaped data","shape", "object type", "data type")
        mydata = [("adl_data:", adl_data.shape, type(adl_data), adl_data.dtype),
                ("adl_labels:", adl_labels.shape ,type(adl_labels), adl_labels.dtype),
                ("adl_names:", adl_names.shape, type(adl_names), adl_names.dtype)]
        print(tabulate(mydata, headers=headers))
    #Split train/test sets, combine or make separate validation set
    #ref for this numpy gymnastics - find index of matching subject to sub_train/sub_test/sub_validate
    #https://numpy.org/doc/stable/reference/generated/numpy.isin.html


    act_num = (adl_labels[:,0])-1 #matlab source was 1 indexed, change to 0 indexed
    sub_num = (adl_labels[:,1]) #subject numbers are in column 1 of labels

    if (not incl_val_group):
        train_index = np.nonzero(np.isin(sub_num, split_subj['train_subj'] + 
                                        split_subj['validation_subj']))
        x_train = adl_data[train_index]
        y_train = act_num[train_index]
    else:
        train_index = np.nonzero(np.isin(sub_num, split_subj['train_subj']))
        x_train = adl_data[train_index]
        y_train = act_num[train_index]

        validation_index = np.nonzero(np.isin(sub_num, split_subj['validation_subj']))
        x_validation = adl_data[validation_index]
        y_validation = act_num[validation_index]

    test_index = np.nonzero(np.isin(sub_num, split_subj['test_subj']))
    x_test = adl_data[test_index]
    y_test = act_num[test_index]

    if (verbose):
        print("x/y_train shape ",x_train.shape,y_train.shape)
        if (incl_val_group):
            print("x/y_validation shape ",x_validation.shape,y_validation.shape)
        print("x/y_test shape  ",x_test.shape,y_test.shape)
    #If selected one-hot encode y_* using keras to_categorical, reference:
    #https://keras.io/api/utils/python_utils/#to_categorical-function and
    #https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    if (one_hot_encode):
        y_train = to_categorical(y_train, num_classes=9)
        if (incl_val_group):
            y_validation = to_categorical(y_validation, num_classes=9)
        y_test = to_categorical(y_test, num_classes=9)
        if (verbose):
            print("After one-hot encoding")
            print("x/y_train shape ",x_train.shape,y_train.shape)
            if (incl_val_group):
                print("x/y_validation shape ",x_validation.shape,y_validation.shape)
            print("x/y_test shape  ",x_test.shape,y_test.shape)
    if (incl_val_group):
        return x_train, y_train, x_validation, y_validation, x_test, y_test
    else:
        return x_train, y_train, x_test, y_test
    
    
#%%


x_train,y_train,x_test, y_test= unimib_load_dataset()

#%%



X_train = x_train.copy()
X_test = x_test.copy()

#%%

print("X train size: ", len(X_train))
print("X test size: ", len(X_test))
print("y train size: ", len(y_train))
print("y test size: ", len(y_test))


y_train_without_encoding = np.argmax(y_train, axis=1)

y_train_without_encoding = np.expand_dims(y_train_without_encoding, axis=1)

print(y_train_without_encoding.shape)

unique_values = np.unique(y_train_without_encoding)

trainX_dict = {value: [] for value in unique_values}

for index, value in enumerate(y_train_without_encoding):
    trainX_dict[value[0]].append(X_train[index])

for value, x_list in trainX_dict.items():
    trainX_dict[value] = np.array(x_list)

for x in unique_values:
    print(f"Label: {x}, Shape: {trainX_dict[x].shape}")
    
    
#%%    


import random


def my_function(data, limit = 2500, threshold = 70):
    total_frames = 0
    for index in data.keys():
        total_frames += data[index].shape[0]
    print(total_frames)

    total_aug = 0

    thes_list = []
    aug_frame = []
    probability = []

    for index in data.keys():
        per_of_data = (data[index].shape[0]/limit)*100
        if per_of_data > threshold:
            thes_list.append(index)

    print(thes_list)

    for index in data.keys():
        if index in thes_list:
            no_of_aug_frame = 0
        else:
            no_of_aug_frame = limit-data[index].shape[0]
        total_aug += no_of_aug_frame
        aug_frame.append(no_of_aug_frame)
        print(f"for Label: {index} no_of_aug_frame is {no_of_aug_frame}")

    for value in aug_frame:
        probability.append(value/total_aug)
    print(probability)

    sum = 0
    aug_dict = {value: [] for value in unique_values}

    for index in range(len(aug_frame)):
        c = 0
        sz = data[index].shape[0]-1
        for i in range(aug_frame[index]):
            x = random.uniform(0, .7)
            if x < probability[index]:
                c += 1
                ran = random.randint(0, sz)
                # print(data[val_list[index]].shape[0], ran)
                aug_data = data[index][ran]
                # print(aug_data.shape)
                aug_data = DA_Jitter(aug_data)
                
                # r=random.randint(10, 150)
                # aug_data= left_shift(aug_data, r)
                # r=random.randint(10, 150)
                # aug_data= right_shift(aug_data, r)
                # r=random.randint(10, 150)
                # aug_data= left_shift(aug_data, r)
                # aug_data = DA_TimeWarp(aug_data)
                # aug_data = DA_Scaling(aug_data)
                
                # aug_data = DA_MagWarp(aug_data)
                #aug_data = DA_MagWarp(DA_Scaling(aug_data))
                # aug_data = DA_Permutation(aug_data)
                #aug_data= DA_RandSampling(aug_data)
                # aug_data= random_mask_data(aug_data,.3)
                #aug_data= gaussian_noise_mask_data(aug_data)
                # aug_data= DA_Rotation(aug_data)
                aug_data= DA_Rotation(aug_data)
                aug_dict[index].append(aug_data)


        sum += c
        print(c)

    print(sum)
    return aug_dict

#%%


aug_dict = my_function(trainX_dict)
#%%
for value, x_list in aug_dict.items():
    aug_dict[value] = np.array(x_list)
    # print(aug_dict[value].shape)

for index in range(len(trainX_dict)):
    if aug_dict[index].shape[0] != 0:
        # print(trainX_dict[index].shape)
        # print(aug_dict[index].shape)
        trainX_dict[index] = np.concatenate((trainX_dict[index], aug_dict[index]), axis = 0)
        print(trainX_dict[index].shape)

new_y_label = []
for index in trainX_dict.keys():
    for i in range(trainX_dict[index].shape[0]):
        new_y_label.append(index)

new_y_label = np.array(new_y_label)

#%%

new_x = np.concatenate((trainX_dict[0], trainX_dict[1], trainX_dict[2], trainX_dict[3], trainX_dict[4], trainX_dict[5], trainX_dict[6], trainX_dict[7], trainX_dict[8]), axis=0)
print(new_x.shape)

#%%

num_rows = new_x.shape[0]

# Create an index array and shuffle it
index_array = np.arange(num_rows)
np.random.shuffle(index_array)

# Use the shuffled index array to shuffle both arrays
new_x = new_x[index_array]
new_y_label = new_y_label[index_array]

#%%

import numpy as np

def one_hot_encode_1d(array):
    # Find the unique values in the input array
    unique_values = np.unique(array)

    # Get the number of unique values
    num_unique_values = len(unique_values)

    # Create an empty 2D array filled with zeros
    one_hot_encoded = np.zeros((array.shape[0], num_unique_values))

    # Set the appropriate elements to 1 based on the input array values
    for i, value in enumerate(array):
        one_hot_encoded[i, np.where(unique_values == value)] = 1

    return one_hot_encoded

new_y = one_hot_encode_1d(new_y_label)

print(new_x.shape)
print(new_y.shape)

#%%

def cnn_model():
    input = Input(shape=(151,3))

    b1 = Conv1D(filters=64, kernel_size=3, activation='relu')(input)
    b1 = Conv1D(filters=32, kernel_size=3, activation='relu')(b1)
    b1 = Dropout(0.3)(b1)
    b1 = MaxPooling1D(pool_size=2)(b1)
    b1 = Flatten()(b1)
    x = Dense(100, activation='relu')(b1)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(9, activation='softmax', name = 'Dense_2')(x)
    model = Model(inputs=input, outputs=x)
    opt = optimizers.RMSprop(lr=0.0001)
    model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model


def v_cnn_model(input_shape=(151, 3), num_classes=9):
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv1D(32, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # Convolutional Layer 2
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # Convolutional Layer 3
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # Convolutional Layer 4
    # model.add(Conv1D(512, kernel_size=3, activation='relu', padding='same'))
    # model.add(Conv1D(512, kernel_size=3, activation='relu', padding='same'))
    # model.add(Conv1D(512, kernel_size=3, activation='relu', padding='same'))
    # model.add(MaxPooling1D(pool_size=2, strides=2))

    # Flatten the output for the fully connected layers
    model.add(Flatten())

    # Fully Connected Layer 1
    model.add(Dense(256, activation='relu'))

    # Fully Connected Layer 2
    model.add(Dense(256, activation='relu'))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    
    opt = optimizers.RMSprop(lr=0.0001)
    model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model
def lstm_model(input_shape=(151,3), num_classes=9):
    model = Sequential()

    # Add an LSTM layer with 128 units and return sequences
    model.add(LSTM(128, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    # # Add another LSTM layer with 64 units and return sequences
    # model.add(LSTM(64, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    # # Add a third LSTM layer with 32 units and return sequences
    # model.add(LSTM(32, return_sequences=True))
    # model.add(Dropout(0.2))
    # model.add(BatchNormalization())

    # Flatten the output to connect to a dense layer
    model.add(Flatten())

    # Add a Dense (fully connected) layer with 128 units and ReLU activation
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(BatchNormalization())

    # Add the final Dense layer with num_classes units and softmax activation
    model.add(Dense(num_classes, activation='softmax'))

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model


def v2_cnn_model(input_shape=(151, 3), num_classes=9):
    model = Sequential()

    # Convolutional Layer 1
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape))
    model.add(Conv1D(64, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # Convolutional Layer 2
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(128, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # Convolutional Layer 3
    model.add(Conv1D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(256, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # Convolutional Layer 4
    model.add(Conv1D(512, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(512, kernel_size=3, activation='relu', padding='same'))
    model.add(Conv1D(512, kernel_size=3, activation='relu', padding='same'))
    model.add(MaxPooling1D(pool_size=2, strides=2))

    # Flatten the output for the fully connected layers
    model.add(Flatten())

    # Fully Connected Layer 1
    model.add(Dense(4096, activation='relu'))

    # Fully Connected Layer 2
    model.add(Dense(4096, activation='relu'))

    # Output Layer
    model.add(Dense(num_classes, activation='softmax'))
    
    opt = optimizers.RMSprop(lr=0.0001)
    model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])

    return model


model=lstm_model()
model.summary()

#%%

#%%
BATCH_SIZE = 32
N_EPOCHS = 50

callback = EarlyStopping(monitor='val_loss', mode = 'auto', patience=20)

# history = model.fit(
#         new_x,new_y,
#         batch_size = BATCH_SIZE,
#         epochs=N_EPOCHS,
#         validation_data=(x_test,y_test))
# epochs=N_EPOCHS, callbacks=[callback],
history = model.fit(
        x_train,y_train,
        batch_size = BATCH_SIZE,
        epochs=N_EPOCHS,
        validation_data=(x_test,y_test))



#%%

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%

# plt.title("Training session's progress over iterations")
# plt.xlabel('Training iteration')
# plt.ylabel('Training Progress(Accuracy values)')
# plt.plot(history.history['accuracy'], label='Train accuracies', color='blue')
# plt.plot(history.history['val_accuracy'], label='Test accuracies', color='red')
# plt.legend()
# plt.grid(True)
# plt.ylim(.7, 1)
# plt.rc('legend', fontsize=8)
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=10)
# plt.show()

# plt.title("Training session's progress over iterations")
# plt.xlabel('Training iteration')
# plt.ylabel('Training progress(Loss values)')
# plt.plot(history.history['loss'], label='Train losses', color='blue')
# plt.plot(history.history['val_loss'], label='Test losses', color='red')
# plt.legend()
# plt.grid(True)
# plt.ylim(0, 1.5)
# plt.rc('legend', fontsize=8)
# plt.rc('xtick', labelsize=10)
# plt.rc('ytick', labelsize=10)
# plt.show()


#%%
from sklearn.metrics import precision_recall_fscore_support
def cal_precision_recall_f1(saved_model, X_test, y_test):
  y_pred_ohe = saved_model.predict(X_test)
  y_pred_labels = np.argmax(y_pred_ohe, axis=1)
  y_true_labels = np.argmax(y_test, axis=1)

  LABELS = ['StandingUpFS','StandingUpFL','Walking','Running','GoingUpS','Jumping','GoingDownS','LyingDownFS','SittingDown']
  labels = [0, 1, 2, 3, 4, 5, 6, 7, 8]
  result=[]
  for label in labels:
    precision, recall, f_score, support = precision_recall_fscore_support(np.array(y_true_labels)==label, np.array(y_pred_labels)==label)
    result.append([LABELS[label], recall[0], recall[1], precision[1], f_score[1], support[1]])
  result_dataframe = pd.DataFrame(result, columns=["label", "specificity", "recall", "precision", "f_score", "support"])
  print(result_dataframe)
  weight = result_dataframe["support"]/result_dataframe["support"].sum()
  result_dataframe_summary = result_dataframe[["specificity", "recall", "precision", "f_score"]].apply(lambda col:np.sum(col*weight), axis=0)
  print('\nAverage according to weight:')
  print(result_dataframe_summary)

cal_precision_recall_f1(model, X_test, y_test)
