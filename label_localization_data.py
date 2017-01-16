import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

SRC_FOLDERS = [os.path.join('data', 'localization', 'train'), os.path.join('data', 'localization', 'test')]
DATA_FILENAME = 'localization_data.csv'
DEFAULT_IMAGE_SIZE = (1440, 1440)
DEFAULT_GRID_SIZE = (24, 24)

def click_handler(fig, cell_size):
	cell_height, cell_width = cell_size

	def onclick(event):
		y = int(event.ydata / cell_height)
		x = int(event.xdata / cell_width)

		min_y = y * cell_height
		min_x = x * cell_height

		plt.gca().add_patch(patches.Rectangle((min_x, min_y), cell_width, cell_height))
		plt.draw()

	return onclick


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
	for filename in os.listdir(src_folder):
		if filename == '.DS_Store':
			continue

		src = os.path.join(src_folder, filename)
		img = cv2.resize(cv2.imread(src), image_size)

		fig = plt.figure(figsize=(10, 10), frameon=False)
		plot = plt.imshow(img)

		plot.axes.get_xaxis().set_visible(False)
		plot.axes.get_yaxis().set_visible(False)
		fig.subplots_adjust(left=0,right=1,bottom=0,top=1)

		cell_size = draw_grid(grid_size, image_size)

		cid = fig.canvas.mpl_connect('button_press_event', click_handler(fig, cell_size))

		fig.tight_layout()
		plt.show()
		print('Showing')
		break


if __name__ == '__main__':
	for folder in SRC_FOLDERS:
		label_localization_data(folder, os.path.join(folder, DATA_FILENAME), DEFAULT_GRID_SIZE, DEFAULT_IMAGE_SIZE)