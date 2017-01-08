import os
import cv2
import numpy as np
from nets.cnn import CNN

TRAIN_SRC_FOLDER = os.path.join('data', 'recognition', 'train')
TEST_SRC_FOLDER = os.path.join('data', 'recognition', 'test')
DEFAULT_IMAGE_SHAPE = (256, 256, 1)

def onehot(arr):
	shape = (len(arr), np.amax(arr) + 1)
	onehot = np.zeros(shape)
	onehot[np.arange(shape[0]), arr] = 1

	return onehot

def read_datastructure(folder, image_shape=DEFAULT_IMAGE_SHAPE):
	print('Reading data from ' + folder)

	X = []
	y = []
	labels = []
	counts = []

	for label in os.listdir(folder):
		src = os.path.join(folder, label)
		if not os.path.isdir(src):
			continue

		label_id = len(labels)
		labels.append(label)
		counts.append(0)
		i = 0
		for filename in os.listdir(src):
			if filename == '.DS_Store':
				continue

			img = cv2.imread(os.path.join(src, filename))
			img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			height, width, _ = image_shape
			img = cv2.resize(img, (height, width))
			X.append(img)
			y.append(label_id)
			counts[label_id] = counts[label_id] + 1
			i += 1
			if i == 1:
				break

	X = np.asarray(X)
	y = np.array(y)
	y = onehot(y)
	counts = np.asarray(counts)
	ratios = counts / np.sum(counts)

	print('Read ' + str(len(X)) + ' images')
	return X, y, labels, ratios

def shuffle_data(X, y):
	idx = np.arange(len(X))
	np.random.shuffle(idx)

	return np.squeeze(X[idx]), np.squeeze(y[idx])

def split_data(X, y, val_split=0.8):
	train_len = int(val_split * len(X))

	return X[:train_len], y[:train_len], X[train_len:], y[train_len:]

def train_recognition_model():
	X, y, labels, ratios = read_datastructure(TRAIN_SRC_FOLDER, DEFAULT_IMAGE_SHAPE)
	X, y = shuffle_data(X, y)
	train_X, train_y, val_X, val_y = split_data(X, y)

	height, width, channels = DEFAULT_IMAGE_SHAPE
	cnn = CNN('Fishes', (height, width, channels), 2, class_weights=(1 - ratios))
	cnn.fit(train_X, train_y, val_X, val_y)

if __name__ == '__main__':
	train_recognition_model()