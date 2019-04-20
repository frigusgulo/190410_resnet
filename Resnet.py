#!/usr/bin/env python

import keras
import keras.datasets as kd
import keras.layers as kl
import keras.regularizers as kr
import keras.models as km
import keras.callbacks as kc
from keras.preprocessing.image import ImageDataGenerator
import scipy.ndimage as snd
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
class Resnet():
	def __init__ (self,data_source,labels,epoch,batch_size,weights,test_image):
		(self.x_train, self.y_train), (self.x_test, self.y_test) = data_source.load_data()
		self.labels = labels
		self.model = None
		self.batch_size = batch_size
		self.epochs = epoch
		self.test_image = cv2.imread(test_image)/255.
		self.test_image = cv2.resize(self.test_image,(32, 32))
		self.weights = weights

	def Resnet(self):
		self.x_train = self.x_train/255.
		self.x_test = self.x_test/255.
		# Convert class vectors to binary class matrices.
		N = len(self.labels)
		self.y_train = keras.utils.to_categorical(self.y_train, N)
		self.y_test = keras.utils.to_categorical(self.y_test, N)
		# Specify the shape of the input image
		input_shape = self.x_train.shape[1:]
		self.inputs = kl.Input(shape=input_shape)
		# First convolution + BN + act
		conv = kl.Conv2D(16,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(self.inputs)
		bn = kl.BatchNormalization()(conv)
		act1 = kl.Activation('relu')(bn)
		# Perform 3 convolution blocks
		for i in range(3):
		    conv = kl.Conv2D(16,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act1)
		    bn = kl.BatchNormalization()(conv)
		    act = kl.Activation('relu')(bn)
		    conv = kl.Conv2D(16,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act)
		    bn = kl.BatchNormalization()(conv)
		    # Skip layer addition
		    skip = kl.add([act1,bn])
		    act1 = kl.Activation('relu')(skip)  
		# Downsampling with strided convolution
		conv = kl.Conv2D(32,(3,3),padding='same',strides=2,kernel_regularizer=kr.l2(1e-4))(act1)
		bn = kl.BatchNormalization()(conv)
		act = kl.Activation('relu')(bn)
		conv = kl.Conv2D(32,(3,3),padding='same',kernel_regularizer=kr.l2(1e-4))(act)
		bn = kl.BatchNormalization()(conv)
		# Downsampling with strided 1x1 convolution
		act1_downsampled = kl.Conv2D(32,(1,1),padding='same',strides=2,kernel_regularizer=kr.l2(1e-4))(act1)
		# Downsampling skip layer
		skip_downsampled = kl.add([act1_downsampled,bn])
		act1 = kl.Activation('relu')(skip_downsampled)
		# This final layer is denoted by a star in the above figure
		for _ in range(2):
		    conv = kl.Conv2D(32, (3, 3), padding="same", kernel_regularizer=kr.l2(1e-4))(act1)
		    bn = kl.BatchNormalization()(conv)
		    act = kl.Activation('relu')(bn)
		    conv = kl.Conv2D(32, (3,3), padding='same', kernel_regularizer=kr.l2(1e-4))(act)
		    bn = kl.BatchNormalization()(conv)
		    # Skip layer addition
		    skip = kl.add([act1,bn])
		    act1 = kl.Activation('relu')(skip)   
		# Downsampling with strided convolution
		conv = kl.Conv2D(64, (3,3), padding='same', strides=2, kernel_regularizer=kr.l2(1e-4))(act1)
		bn = kl.BatchNormalization()(conv)
		act = kl.Activation('relu')(bn)
		conv = kl.Conv2D(64, (3,3), padding='same', kernel_regularizer=kr.l2(1e-4))(act)
		bn = kl.BatchNormalization()(conv)
		# Downsampling with strided 1x1 convolution
		act1_downsampled = kl.Conv2D(64,(1,1),padding='same',strides=2,kernel_regularizer=kr.l2(1e-4))(act1)
		# Downsampling skip layer
		skip_downsampled = kl.add([act1_downsampled,bn])
		act1 = kl.Activation('relu')(skip_downsampled)
		for _ in range(2):
		    conv = kl.Conv2D(64, (3, 3), padding="same", kernel_regularizer=kr.l2(1e-4))(act1)
		    bn = kl.BatchNormalization()(conv)
		    act = kl.Activation('relu')(bn)
		    conv = kl.Conv2D(64, (3,3), padding='same', kernel_regularizer=kr.l2(1e-4))(act)
		    bn = kl.BatchNormalization()(conv)
		    # Skip layer addition
		    skip = kl.add([act1,bn])
		    act1 = kl.Activation('relu')(skip)
		self.act1 = act1 #save for resue in classification
		gap = kl.GlobalAveragePooling2D()(act1)
		bn = kl.BatchNormalization()(gap)
		self.final_dense = kl.Dense(N)(bn)
		self.softmax = kl.Activation('softmax')(self.final_dense)
		self.model = km.Model(inputs=self.inputs,outputs=self.softmax)
		opt = keras.optimizers.rmsprop(lr=0.001,decay=1e-6)
		self.model.compile(loss='categorical_crossentropy',
		              optimizer=opt,
		              metrics=['accuracy'])

	
		# Prepare callbacks for model saving and for learning rate adjustment.
		self.cp_aug = kc.ModelCheckpoint(filepath='./checkpoints_aug',
		                             monitor='val_acc',
		                             verbose=1,
		                             save_best_only=True)

		self.datagen = ImageDataGenerator(
		    featurewise_center=False,  # set input mean to 0 over the dataset
		    samplewise_center=False,  # set each sample mean to 0
		    featurewise_std_normalization=False,  # divide inputs by std of the dataset
		    samplewise_std_normalization=False,  # divide each input by its std
		    zca_whitening=False,  # apply ZCA whitening
		    zca_epsilon=1e-06,  # epsilon for ZCA whitening
		    rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
		    # randomly shift images horizontally (fraction of total width)
		    width_shift_range=0.1,
		    # randomly shift images vertically (fraction of total height)
		    height_shift_range=0.1,
		    shear_range=0.01,  # set range for random shear
		    zoom_range=0.2,  # set range for random zoom
		    channel_shift_range=0.,  # set range for random channel shifts
		    # set mode for filling points outside the input boundaries
		    fill_mode='nearest',
		    cval=0.,  # value used for fill_mode = "constant"
		    horizontal_flip=True,  # randomly flip images
		    vertical_flip=False,  # randomly flip images
		    # set rescaling factor (applied before any other transformation)
		    rescale=None,
		    # set function that will be applied on each input
		    preprocessing_function=None,
		    # image data format, either "channels_first" or "channels_last"
		    data_format=None,
		    # fraction of images reserved for validation (strictly between 0 and 1)
		    validation_split=0.0)

	def model_fit(self):
		self.datagen.fit(self.x_train)

		def lr_schedule(epoch):
			lr = 1e-3
			if epoch > 60:
				lr *= 1e-1
			elif epoch > 120:
				lr *= 1e-2
			return lr
		lr_scheduler = kc.LearningRateScheduler(lr_schedule)
		# Fit the model on the batches generated bydatagen.flow().
		self.model.fit_generator(
		    self.datagen.flow(
		        self.x_train, self.y_train,
		        batch_size=self.batch_size
		    ),
		    steps_per_epoch=len(self.x_train)/self.batch_size,
		    epochs=self.epochs,
		    validation_data=(self.x_test, self.y_test),verbose=1, workers=4,
		    callbacks=[self.cp_aug, lr_scheduler]
		)
	def runAll(self):
		self.Resnet()
		#self.model_fit()

	def ClassAct_Map(self):
		#print("\n\n\n",type(self.act1),"\n\n\n")
		pred_image = self.test_image
		new_model = km.Model(inputs=self.inputs,outputs=(self.act1,self.softmax))
		new_model.load_weights(self.weights)
		x,y,z = self.test_image.shape
		last_conv, probs = new_model.predict(pred_image.reshape((1,x,y,z)))
		print("\nProbs: \n",probs,"\n")
		print("Your Image is a: ",self.labels[np.argmax(probs)])
		fm_0 = last_conv[0,:,:,0]
		fm_0_upscaled = snd.zoom(fm_0,4)
		cv2.imshow('testimage',pred_image)
		cv2.waitKey(0)

if __name__ == "__main__":
	labels = ['airplane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
	Resnet = Resnet(kd.cifar10,labels,int(sys.argv[1]),int(sys.argv[2]),sys.argv[3],sys.argv[4]) #Image(sys.argv[1]))
	Resnet.runAll()
	Resnet.ClassAct_Map()


