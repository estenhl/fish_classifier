import numpy as np

def split_data(X, y, val_split=0.8):
	train_len = int(val_split * len(X))

	return X[:train_len], y[:train_len], X[train_len:], y[train_len:]
