import os
import time

import chess
import chess.pgn
import random
import numpy as np

from state import serialize_game



X_data, y_data = [], []
values = {'0-1':-1, '1/2-1/2':0, '1-0':1}
parsed_dataset = False


def get_dataset(num_of_samples, debug = False):
	for fn in os.listdir("dataset"):	

		lost, won = calculate_num_of_classes()
		if lost >= num_of_samples and won >= num_of_samples:
			break
		parsed_dataset = False
		pgn = open(os.path.join('dataset', fn))

		while lost < num_of_samples and not parsed_dataset:
			parse_pgn_file(pgn, -1)
			lost, won = calculate_num_of_classes()
		while won < num_of_samples and not parsed_dataset:
			parse_pgn_file(pgn, 1)
			lost, won = calculate_num_of_classes()

		print(lost, won)

	lost_data, won_data = get_separated_data()

	X_data = np.concatenate((lost_data[0:num_of_samples], won_data[0:num_of_samples]))


	y_data = np.ones(num_of_samples * 2)
	y_data[0 : num_of_samples] *= 0

	return X_data, y_data

def get_separated_data():
	lost_data, won_data = [], []
	for ind, y in enumerate(y_data):
		if y == -1:
			lost_data.append(X_data[ind])
		else:
			won_data.append(X_data[ind])
	
	return lost_data, won_data

def get_data_from_game(game, board, result):
	for move in game.mainline_moves():
		board.push(move)
		X_data.append(serialize_game(board))
		y_data.append(result)
		board.push(move)

def parse_pgn_file(pgn, result_to_collect, num_of_games = 1000, debug = False):
	for game_num in range(0, num_of_games):
		game = chess.pgn.read_game(pgn)
		
		if game == None:
			parsed_dataset = True
			break

		winning_site = 0


		if(values[game.headers['Result']] == -1):
			winning_site = -1
		elif(values[game.headers['Result']] == 0):
			winning_site = 0
		elif(values[game.headers['Result']] == 1):
			winning_site = 1

		if(winning_site == result_to_collect):
			board = game.board()
			get_data_from_game(game, board, winning_site)
	
	if debug:
		lost, won = calculate_num_of_classes()
		print("Lost: {}, Won: {}".format(lost, won))


def calculate_num_of_classes():
	lost, won = 0, 0
	for i in range(np.asarray(y_data).shape[0]):
		if y_data[i] == -1:
			lost += 1
		else:
			won += 1
		
	return lost, won

if __name__ == "__main__":
	X_train, y_train = get_dataset(num_of_samples = 1000)
	X_test, y_test = get_dataset(num_of_samples = 200)
