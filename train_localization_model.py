import os
import numpy as np
from project import *
from nets import SingleLayerNN
from nets import DeepCNN
from utils.data import split_data
from utils.data import shuffle_data
from utils.data import balance_dataset
from utils.data import parse_localization_data
from utils.data import label_localization_data
from train_recognition_model import train_recognition_model

SRC_FOLDER = os.path.join('data', 'localization', 'train')
DATA_FILE = os.path.join(SRC_FOLDER, 'localization_data.csv')
MODEL_PATH = os.path.join('models', 'localization')
LAYER_NAME = 'conv8:0'


def train_localization_model(recognition_cnn=None, image_shape=DEFAULT_IMAGE_SHAPE, verbose=False):
	if recognition_cnn is None:
		recognition_cnn = train_recognition_model(verbose=verbose)

	gridsize, images, Y = parse_localization_data(SRC_FOLDER, DATA_FILE, image_shape, verbose=verbose)
	features = recognition_cnn.extract_features(images, LAYER_NAME)
	X, y = label_localization_data(features, Y)
	print('X.shape: ' + str(X.shape))
	print('y.shape: ' + str(y.shape))
	X, y = shuffle_data(X, y)
	X, y = balance_dataset(X, y, 2)
	train_X, train_y, val_X, val_y = split_data(X, y)

	cnn = SingleLayerNN('Fishes_localization', 512, 2)
	cnn.fit(train_X, train_y, val_X, val_y, epochs=100)

if __name__ == '__main__':
	train_localization_model(verbose=True)
	
