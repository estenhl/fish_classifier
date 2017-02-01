import numpy as np
from collections import Counter

def balance_dataset(X, y, classes):
	counts = Counter([np.argmax(val) for val in y])
	min_count = min(counts)

	balanced_X = []
	balanced_y = []
	val_counts = np.zeros(classes)
	for i in range(0, len(X)):
		val = np.argmax(y[i])
		if val_counts[val] < min_count:
			balanced_X.append(X[i])
			balanced_y.append(y[i])
			val_counts[val] += 1

	return balanced_X, balanced_y