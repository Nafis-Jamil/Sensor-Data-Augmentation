#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 16:16:30 2023

@author: nafis
"""
#%%
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
from scipy.spatial.distance import euclidean
from dtaidistance import dtw,dtw_ndim

from fastdtw import fastdtw
import random
import pickle
import numpy as np
from scipy.stats import pearsonr
from scipy.interpolate import CubicSpline      # for warping
from transforms3d.axangles import axangle2mat  # for rotation

import matplotlib
#matplotlib.use('Agg')   # Use Agg backend to save figures
import matplotlib.pyplot as plt
#%%



#%%
##################################################
### GLOBAL VARIABLES
##################################################
COLUMN_NAMES = [
    'user',
    'activity',
    'timestamp',
    'x-axis',
    'y-axis',
    'z-axis'
]

LABELS = [
    'Downstairs',
    'Jogging',
    'Sitting',
    'Standing',
    'Upstairs',
    'Walking'
]

DATA_PATH = 'WISDM/WISDM_ar_v1.1_raw.txt'

RANDOM_SEED = 13

# Data preprocessing
TIME_STEP = 100

# Model
N_CLASSES = 6
N_FEATURES = 3  # x-acceleration, y-acceleration, z-acceleration

# Hyperparameters
N_LSTM_LAYERS = 2
N_EPOCHS = 1
L2_LOSS = 0.0015
LEARNING_RATE = 0.0025

# Hyperparameters optimized
SEGMENT_TIME_SIZE = 180
N_HIDDEN_NEURONS = 30
BATCH_SIZE = 32

#%%

#%%
##################################################
### FUNCTIONS
##################################################



    # LOAD DATA
data = pd.read_csv(DATA_PATH, header=None, names=COLUMN_NAMES, on_bad_lines='skip')
data['z-axis'].replace({';': ''}, regex=True, inplace=True)
data = data.dropna()

    # SHOW GRAPH FOR JOGGING
# data[data['activity'] == 'Jogging'][['x-axis']][:50].plot(subplots=True, figsize=(16, 12), title='Jogging')
# plt.xlabel('Timestep')
# plt.ylabel('X acceleration (dg)')

#     # SHOW ACTIVITY GRAPH
# activity_type = data['activity'].value_counts().plot(kind='bar', title='Activity type')
#     #plt.show()

    # DATA PREPROCESSING
data_convoluted = []
labels = []

    # Slide a "SEGMENT_TIME_SIZE" wide window with a step size of "TIME_STEP"
for i in range(0, len(data) - SEGMENT_TIME_SIZE, TIME_STEP):
    x = data['x-axis'].values[i: i + SEGMENT_TIME_SIZE]
    y = data['y-axis'].values[i: i + SEGMENT_TIME_SIZE]
    z = data['z-axis'].values[i: i + SEGMENT_TIME_SIZE]
    data_convoluted.append([x, y, z])

        # Label for a data window is the label that appears most commonly
    label = stats.mode(data['activity'][i: i + SEGMENT_TIME_SIZE])[0][0]
    labels.append(label)

    # Convert to numpy
data_convoluted = np.asarray(data_convoluted, dtype=np.float32).transpose(0, 2, 1)

    # One-hot encoding
labels = np.asarray(pd.get_dummies(labels), dtype=np.float32)
print("Convoluted data shape: ", data_convoluted.shape)
print("Labels shape:", labels.shape)


    # SPLIT INTO TRAINING AND TEST SETS
X_train, X_test, y_train, y_test = train_test_split(data_convoluted, labels, test_size=0.3, random_state=RANDOM_SEED)
print("X train size: ", len(X_train))
print("X test size: ", len(X_test))
print("y train size: ", len(y_train))
print("y test size: ", len(y_test))
#%%
















#%%
    ##### BUILD A MODEL
    # Placeholders
# X = tf.placeholder(tf.float32, [None, SEGMENT_TIME_SIZE, N_FEATURES], name="X")
# y = tf.placeholder(tf.float32, [None, N_CLASSES], name="y")

# y_train = np.expand_dims(y_train, axis=1)

# print(y_train.shape)

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



#%%


def calc_dtw_thresh(N=100):
    dtw_threshold=[]

    for i in range (len(trainX_dict)):
      
        R = trainX_dict[i].shape[0]
        result=0 
        generate_tuples = lambda N, R: [(random.randint(0, R), random.randint(0, R)) for _ in range(N)]
        index_list = generate_tuples(N, R-5)
        for j in range(N):
            in1= index_list[j][0]
            in2= index_list[j][1]
            seg1= trainX_dict[i][in1]
            seg2= trainX_dict[i][in2]
            distance,path=fastdtw(seg1,seg2, dist=euclidean) 
            similarity_score=distance
            result += similarity_score
        dtw_threshold.append((result/N))
    return dtw_threshold
dt_thresh= [0]*6

for i in range(10):
    result= calc_dtw_thresh()
    for j in range(len(dt_thresh)):
        dt_thresh[j]=dt_thresh[j]+result[j]
for i in range(len(dt_thresh)):
    dt_thresh[i]=dt_thresh[i]/10


        

dtw_thresh= calc_dtw_thresh()



#%%
print(len(trainX_dict))

def calc_correlation_thresh(N=100):
    corr_threshold=[]
    for i in range (len(trainX_dict)):
      
        
        R = trainX_dict[i].shape[0]
        
        result=0 
        generate_tuples = lambda N, R: [(random.randint(0, R), random.randint(0, R)) for _ in range(N)]
        index_list = generate_tuples(N, R-5)
        for j in range(N):
            in1= index_list[j][0]
            in2= index_list[j][1]
            seg1= trainX_dict[i][in1]
            seg2= trainX_dict[i][in2]
            seg1= seg1.flatten()
            seg2= seg2.flatten()
            cc, _ = pearsonr(seg1, seg2)
            similarity_score= .5 * (1+cc)    
            result += similarity_score
        corr_threshold.append((result/N)*100)
    return corr_threshold
cc_thresh= [0]*6

for i in range(10):
    result= calc_correlation_thresh()
    for j in range(len(cc_thresh)):
        cc_thresh[j]=cc_thresh[j]+result[j]
for i in range(len(cc_thresh)):
    cc_thresh[i]=cc_thresh[i]/10

print(cc_thresh)



#%%

#**********************************************Augmentation Techniques*****************************
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

#%%


def check_valid_corr(seg, validation_set, indx, margin, N):
    qc= random.sample(range(0, validation_set.shape[0]-1), N)
    result=0
    for i in qc:
        v_seg=validation_set[i]
        seg1= seg.flatten()
        seg2= v_seg.flatten()
        cc, _ = pearsonr(seg1, seg2)
        similarity_score= .5 * (1+cc)    
        result += similarity_score
    score= (result/N)*100
    if(abs(score-cc_thresh[indx])<margin):
        return True
    else:
        return False


#%%

def check_valid_dtw(seg, validation_set, indx, margin, N):
    qc= random.sample(range(0, validation_set.shape[0]-1), N)
    result=0
    for i in qc:
        v_seg=validation_set[i]
        # distance=dtw_ndim.distance(seg, v_seg)
        distance,path=fastdtw(seg,v_seg, dist=euclidean) 
        result += distance
    score= result/N
    error= (abs(dt_thresh[indx]-score))/dt_thresh[indx]
    error= error*100
    print(error)
    if(error<margin):
        return True
    else:
        return False


#%%

def corr_score(seg1, seg2):
    seg1= seg1.flatten()
    seg2= seg2.flatten()
    cc, _ = pearsonr(seg1, seg2)
    similarity_score= .5 * (1+cc)  
    return similarity_score

def dtw_score(seg1, seg2):
    distance,path=fastdtw(seg1,seg2, dist=euclidean) 
    return distance



#%%
save_x=trainX_dict
print(dt_thresh)
#%%
import random


def my_function(data, limit = 3000, threshold = 70):
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
        print('started')
        c = 0
        for i in range(aug_frame[index]):
            x = random.uniform(0, .8)
            if x < probability[index]:
                c += 1
        sz = data[index].shape[0]-1
        while(c>0):
            
            ran = random.randint(0, sz)
            # print(data[val_list[index]].shape[0], ran)
            aug_data = data[index][ran]
            
            og_data= aug_data
          
            aug_data = DA_Jitter(aug_data)
            
            r=random.randint(10, 170)
            aug_data= left_shift(aug_data, r)
            r=random.randint(10, 170)
            aug_data= right_shift(aug_data, r)
            r=random.randint(10, 170)
            aug_data= left_shift(aug_data, r)
            if(check_valid_dtw(og_data,data[index], index, 50, 10)):
                print(c)
                aug_dict[index].append(aug_data)
                c -= 1
        print('finished')
                
            
            
            
            
    return aug_dict
    


#%%


aug_dict = my_function(trainX_dict)

#%%
# import pickle
# with open("aug_dict_30.pkl", "wb") as file:
#     pickle.dump(aug_dict, file)
    
# #%%
aug_dict={}
with open("aug_dict_30.pkl", "rb") as file:
    aug_dict = pickle.load(file)

#%%
for value, x_list in aug_dict.items():
    aug_dict[value] = np.array(x_list)
    # print(aug_dict[value].shape)


print(len(aug_dict[0]))

    
#%%
for index in range(len(trainX_dict)):
    if aug_dict[index].shape[0] != 0:
        # print(trainX_dict[index].shape)
        # print(aug_dict[index].shape)
        trainX_dict[index] = np.concatenate((trainX_dict[index], aug_dict[index]), axis = 0)
        print(trainX_dict[index].shape)
    
    
#%%

new_y_label = []
for index in trainX_dict.keys():
    for i in range(trainX_dict[index].shape[0]):
        new_y_label.append(index)
        
new_y_label = np.array(new_y_label)
        
#%%

new_x = np.concatenate((trainX_dict[0], trainX_dict[1], trainX_dict[2], trainX_dict[3], trainX_dict[4], trainX_dict[5]), axis=0)
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


#%%
def cnn_model():
    input = Input(shape=(180, 3))
  
    b1 = Conv1D(filters=64, kernel_size=3, activation='relu')(input)
    b1 = Conv1D(filters=32, kernel_size=3, activation='relu')(b1)
    b1 = Dropout(0.3)(b1)
    b1 = MaxPooling1D(pool_size=2)(b1)
    b1 = Flatten()(b1)
    x = Dense(100, activation='relu')(b1)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(6, activation='softmax', name = 'Dense_2')(x)
    model = Model(inputs=input, outputs=x)
    opt = optimizers.RMSprop(lr=0.0001)
    model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
  
    return model

#%%

#%%

model=cnn_model()
model.summary()

#%%

#%%

BATCH_SIZE = 32
N_EPOCHS = 100
es= EarlyStopping(monitor='val_loss',
                              patience=5,
                              mode='auto')
history=model.fit(new_x, new_y,
          batch_size=BATCH_SIZE, epochs=N_EPOCHS,
          validation_data=(X_test, y_test))

#%%

#%%

plt.title("Training session's progress over iterations")
plt.xlabel('Training iteration')
plt.ylabel('Training Progress(Accuracy values)')
plt.plot(history.history['accuracy'], label='Train accuracies', color='blue')
plt.plot(history.history['val_accuracy'], label='Test accuracies', color='red')
plt.legend()
plt.grid(True)
plt.ylim(.7, 1)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.show()

#%%

#%%

score = model.evaluate(X_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

#%%

#%%

plt.title("Training session's progress over iterations")
plt.xlabel('Training iteration')
plt.ylabel('Training progress(Loss values)')
plt.plot(history.history['loss'], label='Train losses', color='blue')
plt.plot(history.history['val_loss'], label='Test losses', color='red')
plt.legend()
plt.grid(True)
plt.ylim(0, 1.5)
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.show()

#%%

#%%

import seaborn as sns
y_pred_ohe = model.predict(X_test)
y_pred_labels = np.argmax(y_pred_ohe, axis=1)
y_true_labels = np.argmax(y_test, axis=1)
confusion_matrix = metrics.confusion_matrix(y_true=y_true_labels, y_pred=y_pred_labels)

#%%

#%%

LABELS = [
    'Downstairs',
    'Jogging',
    'Sitting',
    'Standing',
    'Upstairs',
    'Walking'
]

#%%

#%%

sns.set(style='whitegrid', palette='muted', font_scale=0.8)
sns.heatmap(confusion_matrix, xticklabels=LABELS, yticklabels=LABELS, annot=True, fmt="d");
plt.title("Confusion matrix")
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.rc('legend', fontsize=8)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.show();

#%%


#%%

from sklearn.metrics import precision_recall_fscore_support
def cal_precision_recall_f1(saved_model, X_test, y_test):
  y_pred_ohe = saved_model.predict(X_test)
  y_pred_labels = np.argmax(y_pred_ohe, axis=1)
  y_true_labels = np.argmax(y_test, axis=1)
  
  LABELS = [
      'Downstairs',
      'Jogging',
      'Sitting',
      'Standing',
      'Upstairs',
      'Walking'
  ]
  labels = [0, 1, 2, 3, 4, 5]
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

#%%


#%%

cal_precision_recall_f1(model, X_test, y_test)

#%%


#%%



#%%
