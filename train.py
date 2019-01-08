import numpy as np
import os

from model import Model, calc_results


class Trainer:
	def __init__(self, X, y):
		self.X = np.asarray(X)
		self.y = np.asarray(y)
		print('loaded X: {} y: {}'.format(self.X.shape, self.y.shape))
		print(calc_results(self.y))
		self.model = Model(input_shape = self.X.shape[1:], output = 1)



	def train(self):
		self.model.train(self.X, self.y) 

	def get_model(self):
		return self.model

	def save_model(self):
		self.model.save_model()

if __name__ == '__main__':
	X, y = [], []
	for fn in os.listdir("processed"):
		print(fn)
		data = np.load(os.path.join('processed', fn))
		X.extend(data['arr_0'])
		y.extend(data['arr_1'])

	trainer = Trainer(X, y)
	#trainer.get_model().load_weights('model.h5')
	trainer.train()
	trainer.save_model()