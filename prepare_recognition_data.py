import os
import cv2
import random
import numpy as np
from shutil import copyfile

SRC_FOLDER = os.path.join('data', 'raw')
DEST_FOLDER = os.path.join('data', 'recognition')
LABELS = {'positive': ['ALB', 'BET', 'DOL', 'LAG', 'OTHER', 'SHARK', 'YFT'], 'negative': ['NoF']}
TRAIN_SPLIT = 0.8

def fetch_data(src_folder, labels):
	data = []
	for label in LABELS:
		for folder in LABELS[label]:
			i = 0
			src = os.path.join(SRC_FOLDER, folder)
			for filename in os.listdir(src):
				if filename == '.DS_Store':
					continue
					
				data.append((src, filename, label))

	return data

def copy_files(dest, files, target, duplicate=False):
	target_dest = os.path.join(dest, target)
	if not os.path.isdir(target_dest):
		os.mkdir(target_dest)

	for (src, filename, label) in files:
		dest = os.path.join(target_dest, label)

		if not os.path.isdir(dest):
			os.mkdir(dest)

		src_file = os.path.join(src, filename)
		dest_file = os.path.join(dest, filename)

		copyfile(src_file, dest_file)
		if duplicate:
			img = cv2.imread(src_file)
			[prefix, postfix] = filename.split('.')
			cv2.imwrite(os.path.join(dest, prefix + '_hflip.' + postfix), np.fliplr(img))
			cv2.imwrite(os.path.join(dest, prefix + '_vflip.' + postfix), np.flipud(img))
			cv2.imwrite(os.path.join(dest, prefix + '_hvflip.' + postfix), np.flipud(np.fliplr(img)))

if __name__ == '__main__':
	data = fetch_data(SRC_FOLDER, LABELS)
	random.shuffle(data)
	train_len = int(TRAIN_SPLIT * len(data))
	train = data[:train_len]
	test = data[train_len:]

	copy_files(DEST_FOLDER, train, 'train', duplicate=True)
	copy_files(DEST_FOLDER, test, 'test', duplicate=True)


