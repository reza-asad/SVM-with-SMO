import numpy as np
import random
import matplotlib.pyplot as plt

from svm_basic import *


class SVMAdvanced(SVM):
	def __clip(self, m ,n, alpha_m, alpha_n):
		if self.y[m] * self.y[n] == 1:
			gamma = self.params['alpha'][m] + self.params['alpha'][n]
			if gamma <= self.C:
				alpha_n = np.maximum(np.minimum(alpha_n, gamma), 0)
				alpha_m = np.maximum(np.minimum(alpha_m, gamma), 0)
			else:
				alpha_n = np.maximum(np.minimum(alpha_n, self.C), gamma-self.C)
				alpha_m = np.maximum(np.minimum(alpha_m, self.C), gamma-self.C)
		elif self.y[m] * self.y[n] == -1:
			gamma = self.params['alpha'][m] - self.params['alpha'][n]
			if gamma > 0:
				alpha_n = np.maximum(np.minimum(alpha_n, self.C - gamma), 0) 
				alpha_m = np.maximum(np.minimum(alpha_m, self.C), gamma)
			else:
				alpha_n = np.maximum(np.minimum(alpha_n, self.C), -gamma)
				alpha_m = np.maximum(np.minimum(alpha_m, gamma + self.C), 0)
		return alpha_m, alpha_n

	def __solver(self, m, n, epsilon=1e-7):
		# Skip the case that both the indexs for alpha are the same
		if m == n:
			return 0

		# I use the convention that m should be less than n.
		if m > n:
			m, n = n, m

		# Let's save the old parameters
		alpha_n_old = self.params['alpha'][n]
		alpha_m_old = self.params['alpha'][m]
		bias_old = self.params['bias']
		w_old = self.params['w']

		# This is the curvature of the polynomial function in terms of alpha_n
		k = -np.dot(self.X[m,:], self.X[m,:]) - np.dot(self.X[n,:], self.X[n,:]) + \
			2 * np.dot(self.X[n,:], self.X[m,:])

		# Compute the new alphas
		alpha_n = alpha_n_old + self.y[n] * (self.Errors[n] - self.Errors[m]) / (k + epsilon)
		alpha_m = alpha_m_old + self.y[m] * self.y[n] * (alpha_n_old - alpha_n)

		# Clip the computed alpha to satisfy the optimization constraint
		alpha_m, alpha_n = self.__clip(m, n , alpha_m, alpha_n)

		# If there are not enough changes in the alpaha parameters skip the update
		if np.abs(alpha_m - alpha_n) < (alpha_m + alpha_n + epsilon) * epsilon:
			return 0

		# Update the alphas if the change is significant
		self.params['alpha'][m] = alpha_m
		self.params['alpha'][n] = alpha_n

		# Update the weights given the new values of alpha
		self.params['w'] = w_old + (alpha_n - alpha_n_old) * self.y[n] * self.X[n,:] + \
								   (alpha_m - alpha_m_old) * self.y[m] * self.X[m,:]

		# Update the bias
		# Take the bias that enforces the error to be zero for a non-boundary example (support vector).
		# If such bias can't be found take the average of the bias corresponding to each alpha.
		bias_m_new = bias_old - self.Errors[m] - (alpha_m - alpha_m_old) * self.y[m] * \
																		   np.dot(self.X[m,:], self.X[m,:]) - \
					 							 (alpha_n - alpha_n_old) * self.y[n] * \
					 							 						   np.dot(self.X[m,:], self.X[n,:])

		bias_n_new = bias_old - self.Errors[n] - (alpha_n - alpha_n_old) * self.y[n] * \
																		   np.dot(self.X[n,:], self.X[n,:]) - \
												 (alpha_m - alpha_m_old) * self.y[m] * \
												 						   np.dot(self.X[m,:], self.X[n,:])

		if (0 < alpha_m) and (alpha_m < self.C):
			self.params['bias'] = bias_m_new
			self.Errors[m] = 0
			zero_error_example = m
		elif (0 < alpha_n) and (alpha_n < self.C):
			self.params['bias'] = bias_n_new
			self.Errors[n] = 0
			zero_error_example = n
		else:
			self.params['bias'] = 0.5 * (bias_n_new + bias_m_new)
			zero_error_example = None

		# Update the errors
		# Case 2: For boundary examples update the errors using a formula
		#		  similar to the one for computing the bias.
		num_train = len(self.X)
		non_optimized = [i for i in range(num_train) if i != zero_error_example]
		self.Errors[non_optimized] = self.Errors[non_optimized] + self.y[m] * (alpha_m - alpha_m_old) * \
																  np.dot(self.X[m,:], self.X[non_optimized,:].T) + \
									 							  self.y[n] * (alpha_n - alpha_n_old) * \
									 							  np.dot(self.X[n,:], self.X[non_optimized,:].T) + \
									 							  (self.params['bias'] - bias_old)

		# All the updates went through return that the update was successfull
		return 1

	def __choose_second_alpha(m, pos_alpha):
		if len(pos_alpha) > 0:
			# Case1: Choose the second alpha over non boundary examples with max error
			n = np.argmax(np.abs(self.Errors[m] - self.Errors))
			took_step = self.__solver(m, n)
			if took_step:
				return 1

			# Case2: Choose the second alpha over the all non boundary examples 
			#		 starting from a random index.
			start_idx = np.random.randint(len(pos_alpha))
			for i in range(start_idx, len(pos_alpha)):
				took_step = self.__solver(m, i)
				if took_step:
					return 1

		# Case3: Choose the second alpha over all the examples starting from a 
		#		 random index.
		num_train = len(self.X)
		start_idx = np.random.randint(start_idx, num_train)
		for i in range(start_idx, num_train):
			took_step = self.__solver(m, i)
			if took_step:
				return 1

		# Casse 4: No significant change after looking at all other cases
		return 0

	def train(self, print_every=500):
		super(SVMAdvanced, self).train(check_linearity=False, print_every=print_every)









