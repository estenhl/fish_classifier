import cv2

def parse_images(filenames, image_shape):
	images = []

	for filename in filenames:
		images.append(parse_image(filename, image_shape))

	return images

def parse_image(filename, image_shape):
	img = cv2.imread(filename)
	img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	height, width, _ = image_shape
	img = cv2.resize(img, (height, width))

	return img