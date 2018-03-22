import numpy as np
import random


class SVM():
	# Comments later
	def __init__(self, num_train, num_features, reg=0.0, dtype=np.float32):
		self.num_train = num_train
		self.dtype = dtype
		self.reg = reg
		self.params = {}

		# initialize the parameters of the model
		self.params['alpha'] = np.zeros(num_train)
		self.params['w'] = np.ones(num_features)
		self.params['bias'] = 1

	def naive_max_utility(self, X, y, epsilon=1e-3):
		def kkt(X, y, tol=1e-1):
			alpha = self.params['alpha']
			w = self.params['w']
			bias = self.params['bias']

			constrain_neg = 1 - (np.dot(X, w) + bias) * y
			constraint_neg_is_valid = np.max(constrain_neg) < tol

			alpha_pos = [i for i in range(len(alpha)) if alpha[i] > 0]
			constraint_zero = 1 - (np.dot(X[alpha_pos,:], w) + bias) * y[alpha_pos]
			constraint_zero_is_valid = np.sum(np.abs(constraint_zero - tol) > tol) == 0

			# if not constraint_neg_is_valid:
				# print "negative constraint is not valid"
				# print np.max(constrain_neg)
			# if not constraint_zero_is_valid:
				# print "zero constraint is not valid"
				# print constraint_zero
			return constraint_zero_is_valid and constraint_neg_is_valid


		# The utility function is a polynomial of degree 2 in alpha
		# a * alpha^2 + b * alpha + c
		# The solution is -b/(2*a)
		idx = range(self.num_train)
		j = 0
		while not kkt(X,y):
			for i in range(self.num_train-2):
				idx_subset = idx[:i] + idx[i+2:]
				X_subset = X[idx_subset,:]
				y_subset = y[idx_subset]
				alpha_subset = self.params['alpha'][idx_subset]
				x_m = X[i,:]
				y_m = y[i]
				x_n = X[i+1,:]
				y_n = y[i+1]

				si = -np.sum(alpha_subset * y_subset)
				# The coefficient of alpha^2
				a = np.dot(x_m, x_n) - 0.5 * (np.dot(x_m, x_m) + np.dot(x_n, x_n))
				# the coefficient of alpha
				b = np.dot(X_subset, x_m) * y_subset * alpha_subset * y_n - \
					np.dot(X_subset, x_n) * y_subset * alpha_subset * y_n
				b = np.sum(b) + 1 - y_n * y_m  + si * y_n * (np.dot(x_m, x_m) - np.dot(x_m, x_n))

				# Updated alpha_m and alpha_n
				alpha_n = -float(b)/(2*a+epsilon)
				alpha_m = (si - alpha_n * y_n) * y_m
				if alpha_n >= 0 and alpha_m < 0:
					alpha_m = 0
					alpha_n = np.maximum(si * y_n, 0)
				elif alpha_n < 0 and alpha_m >=0:
					alpha_n = 0
					alpha_m = np.maximum(si * y_m, 0)
				elif alpha_n < 0 and alpha_m < 0:
					alpha_n = 0
					alpha_m = 0

				self.params['alpha'][i] = alpha_m
				self.params['alpha'][i+1] = alpha_n
				self.params['w'] = np.sum(X * y[:,np.newaxis] * self.params['alpha'][:,np.newaxis], axis=0)

				positive_X = X[(y==1),:]
				negative_X = X[(y==-1),:]
				self.params['bias'] = -0.5 * (np.max(np.dot(self.params['w'], negative_X.T)) + \
											  np.min(np.dot(self.params['w'], positive_X.T)))
				j += 1 
				# print np.sum(self.params['alpha'] * y), alpha_m * y_m + alpha_n * y_n, si
			if j > 1300:
				break
		return self.params['alpha'], self.params['w'], self.params['bias']

	def predict(self, X):
		y_pred = np.dot(X, self.params['w']) + self.params['bias']
		y_pred[y_pred >= 0] = 1
		y_pred[y_pred < 0] = -1
		return y_pred










