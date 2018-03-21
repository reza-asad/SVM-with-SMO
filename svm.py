import numpy as np
import random


class SVM():
	# Comments later
	def __init__(self, reg=0.0, dtype=np.float32):
		self.dtype = dtype
		self.reg = reg
		self.params = {}

		# initialize the parameters of the model
		self.params['alpha'] = None
		self.params['w'] = None
		self.params['bias'] = None

	def naive_max_utility(self, X, y, epsilon=1e-3):
		# The utility function is a polynomial of degree 2 in alpha
		# a * alpha^2 + b * alpha + c
		# The solution is -b/(2*a)
		num_train  = len(X)
		self.params['alpha'] = np.random.rand(num_train)
		idx = range(num_train)
		i = 0
		while i < 2000:
			m, n = random.sample(idx, 2)
			if m > n:
				m, n = n, m
			idx_subset = idx[:m] + idx[m+1:n] + idx[n+1:]
			X_subset = X[idx_subset,:]
			y_subset = y[idx_subset]
			alpha_subset = self.params['alpha'][idx_subset]
			x_m = X[m,:]
			y_m = y[m]
			x_n = X[n,:]
			y_n = y[n]

			si = -np.sum(alpha_subset * y_subset)
			# The coefficient of alpha^2
			a = np.dot(x_m, x_n) - 0.5 * (np.dot(x_m, x_m) + np.dot(x_n, x_n))
			# the coefficient of alpha
			b = np.dot(X_subset, x_m) * y_subset * alpha_subset * y_n - \
				np.dot(X_subset, x_n) * y_subset * alpha_subset * y_n
			b = np.sum(b) + 1 - y_n * y_m  + si * y_n * (np.dot(x_m, x_m) - np.dot(x_m, x_n))

			# Updated alpha_m and alpha_n
			argmax = -float(b)/(2*a+epsilon)
			if argmax >= 0:
				alpha_n = argmax
			else:
				alpha_n = 0
			alpha_m = np.maximum((si - alpha_n * y_n) * y_m, 0)
			self.params['alpha'][m] = alpha_m
			self.params['alpha'][n] = alpha_n
			i += 1
		self.params['w'] = np.sum(X * y[:,np.newaxis] * self.params['alpha'][:,np.newaxis], axis=0)

		positive_X = X[(y==1).reshape(len(X),),:]
		negative_X = X[(y==-1).reshape(len(X),),:]
		self.params['bias'] = -0.5 * (np.max(np.dot(self.params['w'], negative_X.T)) + \
									  np.min(np.dot(self.params['w'], positive_X.T))) 
		return self.params['alpha'], self.params['w'], self.params['bias']

	def predict(self, X):
		y_pred = np.dot(X, self.params['w']) + self.params['bias']
		y_pred[y_pred >= 0] = 1
		y_pred[y_pred < 0] = -1
		return y_pred










