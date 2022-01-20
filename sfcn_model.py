#import matplotlib.pyplot as plt

import tensorflow as tf

from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, ReLU, AveragePooling3D
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta,Adam
from tensorflow.keras import regularizers


import numpy as np
import pandas as pd 
import os
import SimpleITK as sitk

#from datetime import datetime
import csv
import sys

def rotate(ax, ay, az, atol=1e-8):

    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r=np.zeros((3,3))
    r[0,0] = cz*cy 
    r[0,1] = cz*sy*sx - sz*cx
    r[0,2] = cz*sy*cx+sz*sx     

    r[1,0] = sz*cy 
    r[1,1] = sz*sy*sx + cz*cx 
    r[1,2] = sz*sy*cx - cz*sx

    r[2,0] = -sy   
    r[2,1] = cy*sx             
    r[2,2] = cy*cx

    qs = 0.5*np.sqrt(r[0,0] + r[1,1] + r[2,2] + 1)
    qv = np.zeros(3)

    if np.isclose(qs,0.0,atol): 
        i= np.argmax([r[0,0], r[1,1], r[2,2]])
        j = (i+1)%3
        k = (j+1)%3
        w = np.sqrt(r[i,i] - r[j,j] - r[k,k] + 1)
        qv[i] = 0.5*w
        qv[j] = (r[i,j] + r[j,i])/(2*w)
        qv[k] = (r[i,k] + r[k,i])/(2*w)
    else:
        denom = 4*qs
        qv[0] = (r[2,1] - r[1,2])/denom;
        qv[1] = (r[0,2] - r[2,0])/denom;
        qv[2] = (r[1,0] - r[0,1])/denom;
    return qv

def apply_translation_rotation(thetaX, thetaY, thetaZ, tx, ty, tz, scale, n):

    theta_x_vals = (thetaX[1]-thetaX[0])*np.random.random(n) + thetaX[0]
    theta_y_vals = (thetaY[1]-thetaY[0])*np.random.random(n) + thetaY[0]
    theta_z_vals = (thetaZ[1]-thetaZ[0])*np.random.random(n) + thetaZ[0]
    tx_vals = (tx[1]-tx[0])*np.random.random(n) + tx[0]
    ty_vals = (ty[1]-ty[0])*np.random.random(n) + ty[0]
    tz_vals = (tz[1]-tz[0])*np.random.random(n) + tz[0]
    s_vals = (scale[1]-scale[0])*np.random.random(n) + scale[0]
    res = list(zip(theta_x_vals, theta_y_vals, theta_z_vals, tx_vals, ty_vals, tz_vals, s_vals))
    return [list(rotate(*(p[0:3]))) + list(p[3:7]) for p in res]

# Data generator to load images batch per batch
class DataGenerator(tf.keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, batch_size, dim, shuffle, imageDirectory, covariates,aug):
		'Initialization'
		self.dim = dim
		self.batch_size = batch_size
		self.list_IDs = list_IDs
		self.shuffle = shuffle
		self.imageDirectory = imageDirectory
		self.covariates = covariates
		self.aug = aug
		self.on_epoch_end()

	def __len__(self):
		'Denotes the number of batches per epoch'
		return int(np.floor(len(self.list_IDs) / self.batch_size))

	def __getitem__(self, index):
		'Generate one batch of data'
        # Generate indexes of the batch
		indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
		list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
		X, y = self.__data_generation(list_IDs_temp)

		return X, y

	def on_epoch_end(self):
        #'Updates indexes after each epoch'
		self.indexes = np.arange(len(self.list_IDs))
		if self.shuffle == True:
			np.random.shuffle(self.indexes)

	def __data_generation(self, list_IDs_temp):
       # 'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
		X = np.empty((self.batch_size, *self.dim, 1)) # contains images
		y = np.empty((self.batch_size), dtype=int) # contains covariate

        # Generate data
		for i, ID in enumerate(list_IDs_temp):
            # Store sample
			itk_img = sitk.ReadImage(self.imageDirectory + str(ID) + '.nii.gz')

			if (self.aug == "True"):
				if np.random.uniform() >= .5:
	                #augmentation - rotate and translate the image
					center = np.array(itk_img.TransformContinuousIndexToPhysicalPoint(np.array(itk_img.GetSize())/2.0))
					aug_transform = sitk.Similarity3DTransform()            
					aug_transform.SetCenter(center)
	                
					aug_parameters=apply_translation_rotation(thetaX=(-np.pi/40.0,np.pi/40.0),thetaY=(-np.pi/40.0,np.pi/40.0),thetaZ=(-np.pi/40.0,np.pi/40.0),tx=(-10.0, 10.0),ty=(-10.0, 10.0),tz=(-10.0, 10.0),scale=(1.0,1.0),n=1)
					aug_transform.SetParameters(aug_parameters[0])        
					aug_image=sitk.Resample(itk_img, itk_img, aug_transform)
					img=sitk.GetArrayFromImage(aug_image)
				else:
					img=sitk.GetArrayFromImage(itk_img)
			else:
				img=sitk.GetArrayFromImage(itk_img)

			X[i,] = np.float32(img.reshape(self.dim[0],self.dim[1],self.dim[2],1))

            # Store class
			y[i,] = self.covariates[np.where(self.covariates[:,0]==ID)][0,1]
		return X, y

def sfcn(inputLayer):
    #block 1
    x=Conv3D(filters=32, kernel_size=(3, 3, 3),padding='same',name="conv1")(inputLayer)
    x=BatchNormalization(name="norm1")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),name="maxpool1")(x)
    x=ReLU()(x)

    #block 2
    x=Conv3D(filters=64, kernel_size=(3, 3, 3),padding='same',name="conv2")(x)
    x=BatchNormalization(name="norm2")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),name="maxpool2")(x)
    x=ReLU()(x)

    #block 3
    x=Conv3D(filters=128, kernel_size=(3, 3, 3),padding='same',name="conv3")(x)
    x=BatchNormalization(name="norm3")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),name="maxpool3")(x)
    x=ReLU()(x)

    #block 4
    x=Conv3D(filters=256, kernel_size=(3, 3, 3),padding='same',name="conv4")(x)
    x=BatchNormalization(name="norm4")(x)
    x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),name="maxpool4")(x)
    x=ReLU()(x)

    #block 6
    x=Conv3D(filters=64, kernel_size=(1, 1, 1),padding='same',name="conv6")(x)
    x=BatchNormalization(name="norm6")(x)
    x=ReLU()(x)

    #block 7, different from paper
    x=AveragePooling3D()(x)
    x=Dropout(.5)(x)
    x=Flatten(name="flat1")(x)    
    x=Dense(units=1, activation='linear',name="dense1")(x)

    return x

def train_model(imageDirectory, cov, aug, imagex,imagey,imagez,BATCH_SIZE, IDs_train, IDs_valid):

    savedModelName = #Where to save the mode (.hd5 file)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(savedModelName, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # Generates training and validation batches
    training_generator = DataGenerator(IDs_train, BATCH_SIZE, (imagex,imagey,imagez), True, imageDirectory, cov, aug)
    valid_generator = DataGenerator(IDs_valid, BATCH_SIZE, (imagex,imagey,imagez), True, imageDirectory, cov, False)

    # Compile and train the model
    inputA = Input(shape=(imagex,imagey,imagez,1), name="Input")
    z = sfcn(inputA)
    model = Model([inputA], z)
    model.compile(loss=tf.keras.losses.MeanSquaredError(), optimizer=Adam(lr=0.001, decay=0.0003), metrics=['mae'])

    history = model.fit(training_generator, epochs=300, validation_data=valid_generator,callbacks=[checkpoint_callback],verbose=2)

def test_model(savedModelName,IDs_test,imageDirectory,imagex,imagey,imagez, cov):
    bestModel = tf.keras.models.load_model(savedModelName)
    IDs_test = np.sort(IDs_test)
    testData = DataGenerator(IDs_test, 1, (imagex,imagey,imagez), False, imageDirectory, cov, False)
    testPred = bestModel.predict(testData)
    return testPred


######################### CONFIG PARAM #########################

# csv file containing one column with patients IDs and one with age
covariatesFile = '.csv'
# Where the images are stored
imageDirectory = ''

covName = sys.argv[1] #Indicates the column header of the column containing the age values
aug = sys.argv[2] #True or False to perform data augmentation

# Image size
imagex = 50
imagey = 190
imagez = 155

# Load the csv file
cov = pd.read_csv(covariatesFile).to_numpy()

# Loads the patient IDs
IDs_test = np.load('.npy')
IDs_valid = np.load('.npy')
IDs_train = np.load('.npy')

BATCH_SIZE = 8


