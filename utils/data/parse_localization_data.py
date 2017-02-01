import os
import numpy as np
from utils.data import parse_images

def parse_datafile(datafile, limit=None):
	f = open(datafile, 'r')
	lines = f.readlines()

	grid_description = lines[0].split('x')
	gridsize = (int(grid_description[0]), int(grid_description[1]))

	filenames = []
	Y = np.zeros((len(lines[1:]), gridsize[0], gridsize[1]))

	if limit is None:
		limit = len(lines) - 1

	for i in range(0, min(len(lines) - 1, limit)):
		tokens = lines[i + 1].split(',')
		filenames.append(tokens[0])
		y = [int(x) for x in tokens[1:]]
		Y[i] = np.reshape(y, gridsize)

	return gridsize, filenames, Y

def parse_localization_data(src, datafile, image_shape, limit=None, verbose=False):
	gridsize, filenames, Y = parse_datafile(datafile, limit)
	filenames = [os.path.join(src, filename) for filename in filenames]
	images = parse_images(filenames, image_shape)

	return gridsize, images, Y