import os
from project import *
from nets import SingleLayerCNN
from nets import DeepCNN
from utils.data import split_data
from utils.data import shuffle_data
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
	X, y = shuffle_data(X, y)
	counts = [0, 0]
	for i in y:
		counts[np.argmax(i)] = counts[np.argmax(i)] + 1
	print(str(counts))
	train_X, train_y, val_X, val_y = split_data(X, y)

	cnn = SingleLayerCNN('Fishes_localization', (3, 3, 512), 2)
	cnn.fit(X, y, X, y, epochs=10)

if __name__ == '__main__':
	train_localization_model(verbose=True)
	
