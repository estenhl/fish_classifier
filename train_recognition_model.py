import os
import cv2
import numpy as np
from project import *
from nets import DeepCNN
from utils.data import split_data
from utils.data import shuffle_data
from utils.data import parse_datastructure
from test_recognition_model import test_recognition_model

SRC_FOLDER = os.path.join('data', 'recognition', 'train')
OUTPUT_MODEL_FOLDER = os.path.join('models', 'recognition')

def train_recognition_model(image_shape=DEFAULT_IMAGE_SHAPE, verbose=False):
	if verbose:
		print('Training recognition model')

	X, y, labels, ratios = parse_datastructure(SRC_FOLDER, image_shape, limit=1452, verbose=verbose)
	X, y = shuffle_data(X, y)
	train_X, train_y, val_X, val_y = split_data(X, y)

	height, width, channels = image_shape
	cnn = DeepCNN('Fishes', (height, width, channels), 2, class_weights=(1 - ratios))
	cnn.fit(train_X, train_y, val_X, val_y, epochs=1)

	if not os.path.isdir(OUTPUT_MODEL_FOLDER):
		os.mkdir(OUTPUT_MODEL_FOLDER)

	cnn.save(OUTPUT_MODEL_FOLDER)
	#test_recognition_model(cnn, verbose=verbose)

	return cnn

if __name__ == '__main__':
	train_recognition_model()
