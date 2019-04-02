import numpy as np
import os

from model import Model
from gen_training_data import get_dataset

class Trainer:
	def __init__(self, X, y):
		self.X = np.asarray(X)
		self.y = np.asarray(y)
		print('loaded X: {} y: {}'.format(self.X.shape, self.y.shape))
		print(self.X.shape[1:])
		print(self.X.shape[1:])
		self.model = Model(input_shape = self.X.shape[1:], output = 1)



	def train(self):
		self.model.train(self.X, self.y) 

	def get_model(self):
		return self.model

	def load_weights(self, file_name):
		self.model.load_weights(file_name)

	def save_model(self):
		self.model.save_model()

	def evaluate_model(self, X, y):
		self.model.evaluate(x = X, y = y, batch_size = 512, verbose = 1)


if __name__ == '__main__':
	X_train, y_train = get_dataset(num_of_samples = 10000)
	X_test, y_test = get_dataset(num_of_samples = 2000)
	X_train = np.expand_dims(X_train, axis=-1)
	X_test = np.expand_dims(X_test, axis=-1)

	print(np.asarray(X_train).shape, np.asarray(y_train).shape)

	trainer = Trainer(X_train, y_train)
	#trainer.load_weights('model-06-0.59.hdf5')
	#trainer.get_model().load_weights('model.h5')
	trainer.train()
	trainer.evaluate_model(X_test, y_test)
	trainer.save_model()