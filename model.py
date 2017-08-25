# -*- coding: utf-8 -*-
"""
Created on Tue May  2 21:31:22 2017

@author: mnassef
"""

import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
from keras.regularizers import l2
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

lines = []

#define the input path (data path) here 
input_path = ...

#define the output path (model.h5 , plots) here 
output_path


def preprocess_image(img):
   '''
   Method for processing the image:
       - cropping
       - resizing
       - convert from BGR to YUV
   '''
   # original shape: 160x320x3, input shape for neural net: 66x200x3
   # crop to 90x320x3
   new_img = img[50:140,:,:]
   # resize the image to 66x200x3 (same as nVidia)
   new_img = cv2.resize(new_img,(200, 66), interpolation = cv2.INTER_AREA)
   # convert to YUV color space (as nVidia paper suggests)
   new_img = cv2.cvtColor(new_img, cv2.COLOR_BGR2YUV)
	
   return new_img
   
with open('/input_path/driving_log.csv') as csvfile:
   reader = csv.reader(csvfile)
   for line in reader:
       lines.append(line)
       
images =[]
measurements = []

# this is to edit the current path to fit the floyd path
for line in lines:
   measurement = float(line[3])
   correction = 0.25
   
   # skip it if ~0 speed - not representative of driving behavior
   if float(line[6]) < 0.1 :
           continue
   for i in range(3):
       
       source_path = line[i]
       filename = source_path.split('\\')[-1]
       current_path = '/input_path/IMG/' + filename
       image = cv2.imread(current_path)
       
       images.append(preprocess_image(image))
       
       # Center image
       if (i == 0):
           measurements.append(measurement)
       # Left image
       elif(i == 1):
           measurements.append(measurement + correction)
       # Right image
       else:
           measurements.append(measurement - correction)
           

augmented_images, augmented_measurements = [], []

for image , measurement in zip(images,measurements):
   augmented_images.append(image)
   augmented_measurements.append(measurement)
   augmented_images.append(cv2.flip(image,1))
   augmented_measurements.append(-1*measurement)

   
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# X_train = np.load('/input/features.npy')
# y_train = np.load('/input/labels.npy')

#Model inspired from Nvidia architecture of the end to end network 
#Modifications are made to fit our data and eliminate overfitting like using l2 regularizers    
def my_model():
#    model.add(Dropout(0.2))
    model.add(Convolution2D(24,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution2D(36,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution2D(48,5,5, subsample=(2,2), activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution2D(64,3,3, activation='relu', W_regularizer=l2(0.001)))
    model.add(Convolution2D(64,3,3, activation='relu', W_regularizer=l2(0.001)))
#    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(100, W_regularizer=l2(0.001)))
    model.add(Dense(50, W_regularizer=l2(0.001)))
#    model.add(Dropout(0.2))
    model.add(Dense(10, W_regularizer=l2(0.001)))
    model.add(Dense(1))


model = Sequential()

#normalization layer for the input images
model.add(Lambda (lambda x: (x/255.0) - 0.5, input_shape=(66,200,3)))

my_model()

model.compile(loss='mse', optimizer='adam')

history_object = model.fit(X_train, y_train, validation_split=0.3, shuffle = True, nb_epoch = 10)

model.save('output_path/model.h5')

print ('Model Saved')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig('output_path/MSE.pdf')
