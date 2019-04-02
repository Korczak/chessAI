import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint

import h5py
import random

class Model:
	def __init__(self, input_shape, output=1):
		filepath="model-{epoch:02d}-{val_acc:.2f}.hdf5"
		self.checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
		self.create_model(input_shape, output)
	


	def train(self, X, y):
		XY = list(zip(X,y))
		random.shuffle(XY)
		X,Y = zip(*XY)

		X_train = X[:(int)(0.8*len(X))]
		X_test = X[(int)(0.8*len(X)):]
		y_train = Y[:(int)(0.8*len(Y))]
		y_test = Y[(int)(0.8*len(Y)):]

		X_train = np.asarray(X_train)
		X_test = np.asarray(X_test)
		y_train = np.asarray(y_train)
		y_test = np.asarray(y_test)

		temp_num = 0
		for y in y_train:
			if y == 1:
				temp_num += 1
		print(temp_num)

		self.model.fit(x = X_train, y = y_train,
						batch_size = 32, 
						epochs=200,
						validation_data = (X_test, y_test),
						callbacks = [self.checkpoint]
						)

	def save_model(self):
		self.model.save_weights("model.h5")
		print('saved weights')

	def get_model(self):
		return self.model

	def load_weights(self, file_name):
		self.model.load_weights(file_name)

	def create_model(self, input_shape, output):
		self.model = Sequential()

		self.model.add(Conv2D(64, (3, 3), activation='elu', input_shape=input_shape))
		#self.model.add(MaxPooling2D(pool_size=(2,2)))
		self.model.add(Conv2D(32, (3, 3), activation='elu', padding='same'))
		self.model.add(Conv2D(16, (3, 3), activation='elu', padding='same'))

		self.model.add(Flatten())

		self.model.add((Dense(64, activation='elu')))
		self.model.add(Dense(output, activation='sigmoid'))

		self.model.compile(loss='mse', optimizer='adam', metrics=['acc'])


