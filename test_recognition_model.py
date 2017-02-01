import os
from project import *
from nets.deep_cnn import DeepCNN
from utils.data import parse_datastructure
from utils.eval import validate_prediction

SRC_FOLDER = os.path.join('data', 'recognition', 'test')
MODEL_PATH = os.path.join('models', 'recognition')

def test_recognition_model(cnn=None, image_shape=DEFAULT_IMAGE_SHAPE, verbose=False):
	if verbose:
		print('Testing recognition model')

	X, y, labels, _ = parse_datastructure(SRC_FOLDER, image_shape, limit=50, verbose=verbose)

	height, width, channels = image_shape
	if cnn is None:
		cnn = DeepCNN('Fishes', (height, width, channels), 2)
		cnn.load(MODEL_PATH)
	predictions = cnn.predict(X)

	accuracy, conf_matrix = validate_prediction(y, predictions, labels)
	print('Accuracy: ' + str(accuracy))
	print('Confusion matrix: ')
	print(conf_matrix)

if __name__ == '__main__':
	test_recognition_model()