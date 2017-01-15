import os

SRC_FOLDERS = [os.path.join('data', 'localization', 'train'), os.path.join('data', 'localization', 'test')]
DATA_FILENAME = 'localization_data.csv'

def label_localization_data(src_folder, output_file):

if __name__ == '__main__':
	for folder in SRC_FOLDERS:
		label_localization_data(folder, os.path.join(folder, output_file))