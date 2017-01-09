import os
import cv2
import numpy as np

def onehot(arr):
	shape = (len(arr), np.amax(arr) + 1)
	onehot = np.zeros(shape)
	onehot[np.arange(shape[0]), arr] = 1

	return onehot

def parse_datastructure(folder, image_shape, max=None):
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
			if max is not None and i == max:
				break

	X = np.asarray(X)
	y = np.array(y)
	y = onehot(y)
	counts = np.asarray(counts)
	ratios = counts / np.sum(counts)

	print('Read ' + str(len(X)) + ' images')
	return X, y, labels, ratios