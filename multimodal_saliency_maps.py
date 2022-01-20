import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

import tensorflow as tf

from tensorflow.keras.layers import Activation, Concatenate
from tensorflow.keras.layers import Conv3D, MaxPool3D, Flatten, Dense, ReLU, AveragePooling3D
from tensorflow.keras.layers import Dropout, Input, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adadelta,Adam
from tensorflow.keras import regularizers

import SimpleITK as sitk


# Data generator to load images batch per batch
class DataGenerator(tf.keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, list_IDs, batch_size, dim_t1, dim_tof, shuffle, imageDirectory_t1,imageDirectory_tof, covariates):
		'Initialization'
		self.dim_t1 = dim_t1
		self.dim_tof = dim_tof
		self.batch_size = batch_size
		self.list_IDs = list_IDs
		self.shuffle = shuffle
		self.imageDirectory_t1 = imageDirectory_t1
		self.imageDirectory_tof = imageDirectory_tof
		self.covariates = covariates
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
		X_t1 = np.empty((self.batch_size, *self.dim_t1, 1)) # contains images
		X_tof = np.empty((self.batch_size, *self.dim_tof, 1)) # contains images
		y = np.empty((self.batch_size), dtype=int) # contains covariate

        # Generate data
		for i, ID in enumerate(list_IDs_temp):
            # Store sample
			itk_t1 = sitk.ReadImage(self.imageDirectory_t1 + str(ID) + '.nii.gz')
			itk_tof = sitk.ReadImage(self.imageDirectory_tof + str(ID) + '.nii.gz')

			img_t1=sitk.GetArrayFromImage(itk_t1)
			img_tof=sitk.GetArrayFromImage(itk_tof)

			X_t1[i,] = np.float32(img_t1.reshape(self.dim_t1[0],self.dim_t1[1],self.dim_t1[2],1))
			X_tof[i,] = np.float32(img_tof.reshape(self.dim_tof[0],self.dim_tof[1],self.dim_tof[2],1))

            # Store class
			y[i,] = self.covariates[np.where(self.covariates[:,0]==ID)][0,1]# gets the covariate value for a given ID.
		X = [X_t1,X_tof]
		return X, y

def cnn(inputImage_t1,inputImage_tof):
	# inputImage_t1: input 1
	# inputImage_tof: input 2

	#################################################T1 BRANCHES##################################
	    #block 1
	x=Conv3D(filters=32, kernel_size=(3, 3, 3),padding='same',name="t1_conv1")(inputImage_t1)
	x=BatchNormalization(name="t1_norm1")(x)
	x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),name="t1_maxpool1")(x)
	x=ReLU()(x)

    #block 2
	x=Conv3D(filters=64, kernel_size=(3, 3, 3),padding='same',name="t1_conv2")(x)
	x=BatchNormalization(name="t1_norm2")(x)
	x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),name="t1_maxpool2")(x)
	x=ReLU()(x)

    #block 3
	x=Conv3D(filters=128, kernel_size=(3, 3, 3),padding='same',name="t1_conv3")(x)
	x=BatchNormalization(name="t1_norm3")(x)
	x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),name="t1_maxpool3")(x)
	x=ReLU()(x)

    #block 4
	x=Conv3D(filters=256, kernel_size=(3, 3, 3),padding='same',name="t1_conv4")(x)
	x=BatchNormalization(name="t1_norm4")(x)
	x=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),name="t1_maxpool4")(x)
	x=ReLU()(x)

    #block 6
	x=Conv3D(filters=64, kernel_size=(1, 1, 1),padding='same',name="t1_conv6")(x)
	x=BatchNormalization(name="t1_norm6")(x)
	x=ReLU()(x)

    #block 7, different from paper
	x=AveragePooling3D()(x)
	x=Dropout(.5)(x)
	x=Flatten(name="t1_flat1")(x)    
	dense_t1=Dense(units=1, activation='linear',name="t1_dense1")(x)

	#################################################TOF BRANCHES##################################
	## Branch 1 layer name = #Branch#Block#layer
	# convolutional layers Block 1
	y=Conv3D(filters=32, kernel_size=(3, 3, 3),padding='same',name="tof_conv1")(inputImage_tof)
	y=BatchNormalization(name="tof_norm1")(y)
	y=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),name="tof_maxpool1")(y)
	y=ReLU()(y)

    #block 2
	y=Conv3D(filters=64, kernel_size=(3, 3, 3),padding='same',name="tof_conv2")(y)
	y=BatchNormalization(name="tof_norm2")(y)
	y=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),name="tof_maxpool2")(y)
	y=ReLU()(y)

    #block 3
	y=Conv3D(filters=128, kernel_size=(3, 3, 3),padding='same',name="tof_conv3")(y)
	y=BatchNormalization(name="tof_norm3")(y)
	y=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),name="tof_maxpool3")(y)
	y=ReLU()(y)

    #block 4
	y=Conv3D(filters=256, kernel_size=(3, 3, 3),padding='same',name="tof_conv4")(y)
	y=BatchNormalization(name="tof_norm4")(y)
	y=MaxPool3D(pool_size=(2, 2, 2),strides=(2, 2, 2),name="tof_maxpool4")(y)
	y=ReLU()(y)

    #block 6
	y=Conv3D(filters=64, kernel_size=(1, 1, 1),padding='same',name="tof_conv6")(y)
	y=BatchNormalization(name="tof_norm6")(y)
	y=ReLU()(y)

    #block 7, different from paper
	y=AveragePooling3D()(y)
	y=Dropout(.5)(y)
	y=Flatten(name="tof_flat1")(y)    
	dense_tof=Dense(units=1, activation='linear',name="tof_dense1")(y)
	
	# Bottleneck
	z=Concatenate(axis=1,name="Concat")([dense_t1,dense_tof])

	z=Dense(units=1, activation='linear',name="output_linear")(z)

	return z

def linear_combination_parameters_estimation(valid_pred_model1,valid_pred_model2,y_valid):
	# valid_pred_model1: predictions on the validation data for the model 1 (t1 model)
	# valid_pred_model1: predictions on the validation data for the model 2 (tof model)
	# y_valid: np array containg the outcome (age)

	X=pd.concat([valid_pred_model1,valid_pred_model2],axis=1)
	# add intercept
	X = sm.add_constant(X)
	model = sm.OLS(y_valid,X)
	res = model.fit()
	print(res.summary())
	return res.params

def combine_models(params,imagex_t1,imagey_t1,imagez_t1,imagex_tof,imagey_tof,imagez_tof,imagex_tof,imagey_tof,imagez_tof):
	# params: estimated weights for the linear combination of the two models
	# (imagex_t1,imagey_t1,imagez_t1): size of the t1 images
	# (imagex_tof,imagey_tof,imagez_tof): size of the tof images

	# Pretrained models (using the sfcn)
	tofModel = '.hdf5'
	t1Model = '.hdf5'

	#Initialize last dense layer coefficients using the results from linear_combination_parameters_estimation()
	weights=np.empty((2,1),dtype='float32')
	weights[0,0]=params[1]
	weights[1,0]=params[2]
	initCoeffs = [weights,np.array([params[0]],dtype='float32')]

	# Built the model - possible to add fine tuning here
	inputImage_t1 = Input(shape=(imagex_t1,imagey_t1,imagez_t1,1), name="inputImage_t1")
	inputImage_tof = Input(shape=(imagex_tof,imagey_tof,imagez_tof,1), name="inputImage_tof")
	z = cnn(inputImage_t1,inputImage_tof)
	model = Model([inputImage_t1,inputImage_tof], z)

	# /!\The layer names should match between the pretrained models and the layer names in cnn()
	model.load_weights(tofModel,by_name=True)
	model.load_weights(t1Model,by_name=True)
	model.layers[-1].set_weights(initCoeffs)

	return model


def smooth_grad(model,x_test,y_test,num_samples):
	# num_samples: Number of noisy images to generate per test dataset
	# model: Model used to generate the maps
	# x_test: np array containing the different inputs
	# y_test: np array containg the outcome (age)

	# Apply some noise on both inputs
	repeated_images1 = np.repeat(x_test[0], num_samples, axis=0)
	noise1 = np.random.normal(0, 0.1, repeated_images1.shape).astype(np.float32)
	noisy_images1 = tf.math.add(repeated_images1,noise1)
	print(noisy_images1.shape)

	repeated_images2 = np.repeat(x_test[1], num_samples, axis=0)
	noise2 = np.random.normal(0, 0.1, repeated_images2.shape).astype(np.float32)
	noisy_images2 = tf.math.add(repeated_images2,noise2)
	print(noisy_images2.shape)

	# Passes the inputs through the model
	with tf.GradientTape() as tape:
		inputs = [tf.cast(noisy_images1, tf.float32),tf.cast(noisy_images2, tf.float32)]
		tape.watch(inputs)
		predictions = model(inputs)
		# computes the loss between the predicted and true value
		loss=tf.keras.losses.mean_squared_error(y_test, predictions)
		print(predictions)
		print(loss)
	grads = tape.gradient(loss, inputs)
	grads_per_image1 = tf.reshape(grads[0], (-1, num_samples, *grads[0].shape[1:]))
	# Takes absolute values of gradients - we are interested in magnitude only
	grads_per_image1= tf.abs(grads_per_image1)
	# Average gradients over the num_samples noisy inputs
	averaged_grads1 = tf.reduce_mean(grads_per_image1, axis=1)

	grads_per_image2 = tf.reshape(grads[1], (-1, num_samples, *grads[1].shape[1:]))
	# Takes absolute values of gradients - we are interested in magnitude only
	grads_per_image2 = tf.abs(grads_per_image2)
	# Average gradients over the num_samples noisy inputs
	averaged_grads2 = tf.reduce_mean(grads_per_image2, axis=1)

	grayscale_gradients1=tf.squeeze(tf.cast(averaged_grads1, tf.float32)) 

	grayscale_gradients2=tf.squeeze(tf.cast(averaged_grads2, tf.float32)) 


	return [grads_per_image1,grads_per_image2]


def generate_saliency_maps(model,x_test,y_test,num_samples,IDs_test):
	# num_samples: Number of noisy images to generate per test dataset
	# model: Model used to generate the maps
	# x_test: np array containing the different inputs
	# y_test: np array containg the outcome (age)
	# IDs_test: IDs of the test data

	# random t1 and tof image used to copy the header and correctly save the saliency maps
	exampleImage_itk_t1=sitk.ReadImage('.nii.gz')
	exampleImage_itk_tof=sitk.ReadImage('.nii.gz')

	for index, ids in enumerate(IDs_test):
		print(ids)
		# get saliency maps from smooth_grad
		res = smooth_grad(model,x_test,y_test,num_samples)
		res_t1 = res[0]
		res_tof = res[1]
		# save saliency maps
		newIm_t1=sitk.GetImageFromArray(res_t1)
		newIm_t1.CopyInformation(exampleImage_itk_t1)
		sitk.WriteImage(newIm_t1,outputDir + 't1_' + str(ids) + '.nii.gz')

		newIm_tof=sitk.GetImageFromArray(res_tof)
		newIm_tof.CopyInformation(exampleImage_itk_tof)
		sitk.WriteImage(newIm_tof,outputDir + 'tof_' + str(ids) + '.nii.gz')


######################### CONFIG PARAM #########################


# csv file containing one column with patients IDs and one with age
covariatesFile = '.csv'
# Where the images are stored
imageDirectory_t1 = ''
imageDirectory_tof = ''
# Where the saliency maps are stored
outputDir = ''

# Image size
imagex_t1 = 160
imagey_t1 = 192
imagez_t1 = 144

imagex_tof = 50
imagey_tof = 190
imagez_tof = 155

# Load the csv file
cov = pd.read_csv(covariatesFile).to_numpy()

# Loads the patient IDs
IDs_test = np.sort(np.load('.npy'))
IDs_valid = np.sort(np.load('.npy'))
IDs_train = np.sort(np.load('.npy'))

# test predictions generated by the sfcn
test_t1 = pd.DataFrame(np.load('.npy'),columns=['T1'])
test_tof = pd.DataFrame(np.load('.npy'),columns=['TOF'])

# validation predictions generated by the sfcn
valid_t1 = pd.DataFrame(np.load('.npy'),columns=['T1'])
valid_tof = pd.DataFrame(np.load('.npy'),columns=['TOF'])

# outcome (true age) for the validation data - useful in linear_combination_parameters_estimation()
y_valid = cov[cov['Index'].isin(IDs_valid)]
y_valid = y_valid.sort_values(by=['Index'])
y_valid = y_valid.loc[:,'AGE'].to_numpy()