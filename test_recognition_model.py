import os
from nets.cnn import CNN
from utils.data import parse_datastructure
from utils.eval import validate_prediction

TEST_SRC_FOLDER = os.path.join('data', 'recognition', 'train')
MODEL_PATH = os.path.join('models', 'recognition', 'model.ckpt')
DEFAULT_IMAGE_SHAPE = (128, 128, 1)

def test_recognition_model():
	X, y, labels, _ = parse_datastructure(TEST_SRC_FOLDER, DEFAULT_IMAGE_SHAPE, max=100)

	height, width, channels = DEFAULT_IMAGE_SHAPE
	cnn = CNN('Fishes', (height, width, channels), 2)
	cnn.load(MODEL_PATH)
	predictions = cnn.predict(X, y)

	accuracy, conf_matrix = validate_prediction(y, predictions, labels)
	print('Accuracy: ' + str(accuracy))
	print('Confusion matrix: ')
	print(conf_matrix)

if __name__ == '__main__':
	test_recognition_model()