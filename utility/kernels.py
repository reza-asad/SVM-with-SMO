import numpy as np
# A set of kernels supported by my SVM package

def gaussian_kernel(X, Z, kernel_params):
	sigma = kernel_params['sigma']
	if len(X.shape) > 1:
		X_norm = np.sum(X * X, axis=1)[:,np.newaxis]
	else:
		X_norm = np.sum(X * X)
	if len(Z.shape) > 1:
		Z_NORM = np.sum(Z * Z, axis=1)[:,np.newaxis].T
	else:
		Z_NORM = np.sum(Z * Z)
	distance = np.exp((-X_norm - Z_NORM + 2 * X_norm * Z_NORM) / (2 * sigma**2))
	return distance

def linear_kernel(X, Z, kernel_params):
	pass