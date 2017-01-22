import os
import cv2
import sys
import signal
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import PatchCollection

SRC_FOLDERS = [os.path.join('data', 'localization', 'train'), os.path.join('data', 'localization', 'test')]
DATA_FILENAME = 'localization_data.csv'
DEFAULT_IMAGE_SIZE = (1440, 1440)
DEFAULT_GRID_SIZE = (24, 24)

class Monitor:
	def __init__(self):
		self.run = True

def click_handler(monitor, ax, cell_size, patches):
	cell_height, cell_width = cell_size

	def onclick(event):
		button = event.button

		if event.button == 3:
			plt.close()
			return
		elif event.button == 2:
			plt.close()
			monitor.run = False
			return

		y = int(event.ydata / cell_height)
		x = int(event.xdata / cell_width)

		min_y = y * cell_height
		min_x = x * cell_width

		patch = patches[y][x]
		if patch is None:
			patches[y][x] = mpatches.Rectangle((min_x, min_y), cell_width, cell_height, alpha=0.5, color='green')
			ax.add_patch(patches[y][x])
		else:
			patches[y][x].remove()
			patches[y][x] = None
		plt.draw()

	return onclick
def parse_existing(filename):
	existing = []

	if os.path.isfile(filename):
		f = open(filename, 'r')
		for line in f.readlines()[1:]:
			existing.append(line.split(',')[0].strip())

		f.close()

	return existing

def draw_grid(grid_size, image_size):
	grid_height, grid_width = grid_size
	image_height, image_width = image_size
	cell_height = image_height / grid_height
	cell_width = image_width / grid_width

	for i in range(0, grid_height):
		plt.plot([0, image_width], [i * cell_height, i * cell_height], color='black')

	for i in range(0, grid_width):
		plt.plot([i * cell_width, i * cell_width], [0, image_height], color='black')

	return (cell_height, cell_width)

def label_localization_data(src_folder, output_file, grid_size, image_size):
	labels = {}
	existing = parse_existing(output_file)
	monitor = Monitor()

	for filename in os.listdir(src_folder):
		if filename == '.DS_Store' or filename in existing:
			continue

		# Skip flipped images
		if len(filename.split('_')) > 2:
			continue

		src = os.path.join(src_folder, filename)
		img = cv2.resize(cv2.imread(src), image_size)

		fig = plt.figure(figsize=(10, 10), frameon=False)
		plot = plt.imshow(img)

		plot.axes.get_xaxis().set_visible(False)
		plot.axes.get_yaxis().set_visible(False)
		fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

		grid_height, grid_width = grid_size
		cell_size = draw_grid(grid_size, image_size)
		patches = np.empty(grid_size, dtype=object)

		fig.canvas.mpl_connect('button_press_event', click_handler(monitor, plt.gca(), cell_size, patches))

		fig.tight_layout()
		plt.show()

		labels[filename] = np.asarray([0 if x is None else 1 for x in patches.flatten()])

		# Add labels for flipped images
		prefix, suffix = filename.split('.')

		infixes = [
			('_hflip', lambda x: np.fliplr(x).flatten()),
			('_vflip', lambda x: np.flipud(x).flatten()),
			('_hvflip', lambda x: np.fliplr(np.flipud(x)).flatten())
		]

		for (infix, f) in infixes:
			filename = prefix + infix + '.' + suffix
			if os.path.isfile(os.path.join(src_folder, filename)):
				labels[filename] = [0 if x is None else 1 for x in f(patches)]
		
		if not monitor.run:
			break

	if os.path.isfile(output_file):
		out = open(output_file, 'a')
	else:
		out = open(output_file, 'w')
		out.write('x'.join([str(x) for x in grid_size]) + '\n')

	for label in labels:
		out.write(label + ',' + ','.join([str(x) for x in labels[label]]) + '\n')

	out.close()

if __name__ == '__main__':
	for folder in SRC_FOLDERS:
		label_localization_data(folder, os.path.join(folder, DATA_FILENAME), DEFAULT_GRID_SIZE, DEFAULT_IMAGE_SIZE)