import os
import cv2
import numpy as np
from nets.cnn import CNN
from utils.data import split_data
from utils.data import shuffle_data
from utils.data import parse_datastructure
from test_recognition_model import test_recognition_model

TRAIN_SRC_FOLDER = os.path.join('data', 'recognition', 'train')
OUTPUT_MODEL_FOLDER = os.path.join('models', 'recognition')
OUTPUT_MODEL_FILENAME = os.path.join(OUTPUT_MODEL_FOLDER, 'model.ckpt')
DEFAULT_IMAGE_SHAPE = (24, 24, 1)

def train_recognition_model():
	X, y, labels, ratios = parse_datastructure(TRAIN_SRC_FOLDER, DEFAULT_IMAGE_SHAPE, max=80)
	X, y = shuffle_data(X, y)
	train_X, train_y, val_X, val_y = split_data(X, y)

	height, width, channels = DEFAULT_IMAGE_SHAPE
	cnn = CNN('Fishes', (height, width, channels), 2, class_weights=(1 - ratios))
	cnn.fit(train_X, train_y, val_X[0:10], val_y[0:10], epochs=1)

	if not os.path.isdir(OUTPUT_MODEL_FOLDER):
		os.mkdir(OUTPUT_MODEL_FOLDER)

	cnn.save(OUTPUT_MODEL_FILENAME)

	test_recognition_model()

if __name__ == '__main__':
	train_recognition_model()