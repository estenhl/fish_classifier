import os
from nets.deep_cnn import DeepCNN
from utils.data import parse_datastructure
from utils.eval import validate_prediction

SRC_FOLDER = os.path.join('data', 'recognition', 'test')
MODEL_PATH = os.path.join('models', 'recognition')
DEFAULT_IMAGE_SHAPE = (288, 288, 1)

def test_recognition_model(cnn=None, verbose=False):
	if verbose:
		print('Testing recognition model')

	X, y, labels, _ = parse_datastructure(SRC_FOLDER, DEFAULT_IMAGE_SHAPE, verbose=verbose)

	height, width, channels = DEFAULT_IMAGE_SHAPE
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