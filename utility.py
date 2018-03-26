import numpy as np


def kkt(X, y, alpha, w, bias, tol=1e-2 * 8):
	constrain_neg = 1 - (np.dot(X, w) + bias) * y
	constraint_neg_is_valid = np.max(constrain_neg) < tol

	alpha_pos = [i for i in range(len(alpha)) if alpha[i] > 0]
	constraint_zero_is_valid = True
	if len(alpha_pos) > 0:
		constraint_zero = 1 - (np.dot(X[alpha_pos,:], w) + bias) * y[alpha_pos]
		constraint_zero_is_valid = np.sum(np.abs(constraint_zero - tol) > tol) == 0

	return constraint_zero_is_valid and constraint_neg_is_valid
