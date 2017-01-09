import os
from nets.cnn import CNN
from utils.data import parse_datastructure

TEST_SRC_FOLDER = os.path.join('data', 'recognition', 'test')
MODEL_PATH = os.path.join('models', 'recognition', 'model.ckpt')
DEFAULT_IMAGE_SHAPE = (128, 128, 1)

def test_recognition_model():
	X, y, _, _ = parse_datastructure(TEST_SRC_FOLDER, DEFAULT_IMAGE_SHAPE, max=10)

	height, width, channels = DEFAULT_IMAGE_SHAPE
	cnn = CNN('Fishes', (height, width, channels), 2)
	cnn.load(MODEL_PATH)
	predictions = cnn.predict(X)

	for prediction in predictions:
		print(prediction)

if __name__ == '__main__':
	test_recognition_model()