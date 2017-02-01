import os
import cv2
import numpy as np
from .parse_image import parse_image

def onehot(arr):
	shape = (len(arr), np.amax(arr) + 1)
	onehot = np.zeros(shape)
	onehot[np.arange(shape[0]), arr] = 1

	return onehot

def parse_datastructure(folder, image_shape, limit=None, verbose=False):
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
			if not filename.endswith('.jpg'):
				continue

			img = parse_image(os.path.join(src, filename), image_shape)
			X.append(img)
			y.append(label_id)
			counts[label_id] = counts[label_id] + 1
			i += 1
			if limit is not None and i == limit:
				break
		if verbose:
			print('Read ' + str(i) + ' images from ' + src)

	X = np.asarray(X)
	y = np.array(y)
	y = onehot(y)
	counts = np.asarray(counts)
	ratios = counts / np.sum(counts)

	if verbose:
		print('Read ' + str(len(X)) + ' total images')
		
	return X, y, labels, ratios