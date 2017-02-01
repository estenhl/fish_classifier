import os
import itertools
import numpy as np
from .onehot import onehot

def is_valid_index(features, index):
	y, x = index

	return y >= 0 and y < len(features) and x >= 0 and x < len(features[y])

def extract_matrix(feature, indices, grid_size):
	_, _, depth = feature.shape
	matrix = np.zeros((grid_size + (depth,)))

	grid_height, grid_width = grid_size
	for i in range(0, grid_height):
		for j in range(0, grid_width):
			index = indices[(i * grid_height) + j]
			if is_valid_index(feature, index):
				matrix[i][j] = feature[index[0]][index[1]]

	return matrix

def label_localization_data(features, labels, grid_size=(3, 3)):
	indices = np.asarray([-1, 0, 1])
	import random

	num, height, width, depth = features.shape
	grid_height, grid_width = grid_size
	X = np.zeros((len(features) * height * width, grid_height, grid_width, depth))
	y = np.zeros(len(features) * height * width, dtype=int)
	for cnt in range(0, num):
		feature = features[cnt]
		for i in range(0, len(feature)):
			for j in range(0, len(feature[i])):
				vertical_indices = i + indices
				horizontal_indices = j + indices
				X[(cnt * height * width) + (i * height) + j] = extract_matrix(feature, list(itertools.product(vertical_indices, horizontal_indices)), grid_size)
				y[(cnt * height * width) + (i * height) + j] = random.randint(0, 1)#labels[cnt][height][width].astype(int)

	return X, onehot(y)
