import chess
import numpy as np
import random

from state import State
from model import Model

class Game:
	def __init__(self, model):
		self.board = chess.Board()
		self.model = model

	def start_game(self):
		self.board = chess.Board()
		while not self.board.is_game_over():
			print(self.board)
			print()
			print(list(self.board.legal_moves))
			human_move = self.get_human_move()
			#self.board.push(self.get_random_move())
			self.board.push(self.get_best_move())
			print()
		
		print(self.board)
		print('Game ended')
		print(self.board.result())


	def get_random_move(self):
	    num_moves = self.board.legal_moves.count()
	    choice = random.randint(0, num_moves - 1)
	    return list(self.board.legal_moves)[choice]

	def get_human_move(self):
		while True:
			move = input()
			move = chess.Move.from_uci(move)
			if move in self.board.legal_moves:
				return move 

	def get_best_move(self):
		possible_moves = list(self.board.legal_moves)

		predictions = []

		for i, move in enumerate(possible_moves):

			board_temp = self.board.copy()
			board_temp.push(move)
			X = np.asarray(State(board_temp).serialize())
			X = np.expand_dims(X, axis=-1)
			#print(np.asarray(X).shape)

			predictions.append(self.model.predict(X)[0][0])

		return possible_moves[np.argmax(predictions)]

if __name__ == '__main__':
	model = Model()
	model.load_weights('model-04-0.98.h5')

	game = Game(model.get_model())
	game.start_game()


