import numpy as np
from .onehot import onehot

def label_localization_data(features, labels, grid_size=(3, 3)):
	num, height, width, depth = features.shape
	X = np.reshape(features, (num * height * width, depth))
	y = labels.flatten()

	return X, onehot(y)
