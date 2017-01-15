import os
import random
import shutil

SRC_FOLDERS = [os.path.join('data', 'recognition', 'train', 'positive'), 
			os.path.join('data', 'recognition', 'test', 'positive')]
DEST_FOLDER = os.path.join('data', 'localization')
TRAIN_SPLIT = 0.8

def copy_files(src_folders, dest_folder, train_split=1.0):
	if not os.path.isdir(dest_folder):
		os.mkdir(dest_folder)

	for src_folder in src_folders:
		for filename in os.listdir(src_folder):
			if filename == '.DS_Store':
				continue
			
			src_file = os.path.join(src_folder, filename)

			if random.random() <= TRAIN_SPLIT:
				target = 'train'
			else:
				target = 'test'

			dest = os.path.join(dest_folder, target)

			if not os.path.isdir(dest):
				os.mkdir(dest)

			dest_file = os.path.join(dest, filename)
			shutil.copyfile(src_file, dest_file)


if __name__ == '__main__':
	copy_files(SRC_FOLDERS, DEST_FOLDER, TRAIN_SPLIT)