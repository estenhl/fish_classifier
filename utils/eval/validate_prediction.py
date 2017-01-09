import numpy as np
import pandas as pd

def validate_prediction(true_y, pred_y, labels):
	conf_matrix = np.zeros((len(labels), len(labels)))

	for i in range(0, len(true_y)):
		conf_matrix[np.argmax(true_y[i])][np.argmax(pred_y[i])] += 1

	accuracy = np.trace(conf_matrix) / len(true_y)

	return accuracy, pd.DataFrame(conf_matrix, index=labels, columns=labels)