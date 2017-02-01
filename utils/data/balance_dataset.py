import numpy as np
from collections import Counter

def balance_dataset(X, y, classes):
	counts = Counter([np.argmax(val) for val in y])
	print('Counts: ' + str(counts))
	min_count = min([counts[x] for x in counts])
	print('Min count: ' + str(min_count))

	balanced_X = []
	balanced_y = []
	val_counts = np.zeros(classes)
	print('Val counts: ' + str(val_counts))
	for i in range(0, len(X)):
		val = np.argmax(y[i])
		print('Found val: ' + str(val))
		if val_counts[val] < min_count:
			balanced_X.append(X[i])
			balanced_y.append(y[i])
			val_counts[val] += 1

	return balanced_X, balanced_y