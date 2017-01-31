import os
from nets.deep_cnn import DeepCNN
from train_recognition_model import train_recognition_model

SRC_FOLDER = os.path.join('data', 'localization', 'train')
DATA_FILE = os.path.join(SRC_FOLDER, 'localization_data.csv')
MODEL_PATH = os.path.join('models', 'localization')
DEFAULT_IMAGE_SHAPE = (288, 288, 1)
LAYER_NAME = 'conv8'


def train_localization_model(recognition_cnn=None, verbose=False):
	if recognition_cnn is None:
		recognition_cnn = train_recognition_model(verbose=verbose)

	X, _, _, _ = parse_datastructure(SRC_FOLDER, DEFAULT_IMAGE_SHAPE, verbose=verbose)
	features = recognition_cnn.extract_features()

if __name__ == '__main__':
	train_recognition_model(verbose=True)

