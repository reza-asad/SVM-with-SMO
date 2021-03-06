import numpy as np
import random
import matplotlib.pyplot as plt

from svm_basic import *
import config as cfg


class SVMAdvanced():
	def __init__(self, X, y, C=float('inf'), kernel_choice=None, kernel_function=None, 
				 dtype=np.float64, verbose=False, kernel_params={}):
		self.X = X
		self.y = y

		self.dtype = dtype
		self.verbose = verbose
		self.params = {}
		self.objective_func_values = []

		# initialize the parameters of the model
		N, D = X.shape
		self.params['alpha'] = np.zeros(N)
		self.params['w'] = np.zeros(D)
		self.params['bias'] = 0

		self.kernel_choice = kernel_choice
		self.kernel_function = kernel_function
		self.kernel_params = kernel_params

		self.C = C
		self.Errors = self.predict(self.X) - self.y


	def __evaluate_objective_function(self):
		temp = self.y * self.params['alpha']
		result = np.sum(self.params['alpha']) - 0.5 * np.dot(np.dot(temp, 
														    		self.__kernel(self.X,
														    					  self.X)
														    		),
															temp)
		return result

	def __kernel(self, X, Z):
		# Case1: If kernel_choice is Gaussian or Linear use them
		# Case 2: Try to use the kernel function provided by the user.
		# Case 3: Use the dot product. 
		if self.kernel_choice not in cfg.SUPPORTED_KERNELS:
			if self.kernel_function is None:
				return np.dot(X, Z.T)
			else:
				return self.kernel_function(X, Z, self.kernel_params)
		else:
			kernel = cfg.SUPPORTED_KERNELS[self.kernel_choice]
			return kernel(X, Z, self.kernel_params)


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

	def __solver(self, m, n, epsilon=1e-5):
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
		k = -self.__kernel(self.X[m,:], self.X[m,:]) - self.__kernel(self.X[n,:], self.X[n,:]) + \
			2 * self.__kernel(self.X[n,:], self.X[m,:])

		# Compute the new alphas
		alpha_n = alpha_n_old + self.y[n] * (self.Errors[n] - self.Errors[m]) / (k + epsilon)
		alpha_m = alpha_m_old + self.y[m] * self.y[n] * (alpha_n_old - alpha_n)

		# Clip the computed alpha to satisfy the optimization constraint
		alpha_m, alpha_n = self.__clip(m, n , alpha_m, alpha_n)

		# If there are not enough changes in the alpaha parameters skip the update
		if np.abs(alpha_n - alpha_n_old) < (alpha_n + alpha_n_old + epsilon) * epsilon:
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
																		   self.__kernel(self.X[m,:], self.X[m,:]) - \
					 							 (alpha_n - alpha_n_old) * self.y[n] * \
					 							 						   self.__kernel(self.X[m,:], self.X[n,:])

		bias_n_new = bias_old - self.Errors[n] - (alpha_n - alpha_n_old) * self.y[n] * \
																		   self.__kernel(self.X[n,:], self.X[n,:]) - \
												 (alpha_m - alpha_m_old) * self.y[m] * \
												 						   self.__kernel(self.X[m,:], self.X[n,:])

		if (0 < alpha_m) and (alpha_m < self.C):
			self.params['bias'] = bias_m_new
			self.Errors[m] = 0
			zero_error_index = m
		elif (0 < alpha_n) and (alpha_n < self.C):
			self.params['bias'] = bias_n_new
			self.Errors[n] = 0
			zero_error_index = n
		else:
			self.params['bias'] = 0.5 * (bias_n_new + bias_m_new)
			zero_error_index = None

		# Update the errors
		# Case 2: For the rest of alphas update the errors using a formula
		#		  similar to the one for computing the bias.
		num_train = len(self.X)
		non_optimized = [i for i in range(num_train) if i != zero_error_index] 
		self.Errors[non_optimized] = self.Errors[non_optimized] + self.y[m] * (alpha_m - alpha_m_old) * \
																  self.__kernel(self.X[m,:], self.X[non_optimized,:]) + \
									 							  self.y[n] * (alpha_n - alpha_n_old) * \
									 							  self.__kernel(self.X[n,:], self.X[non_optimized,:]) + \
									 							  (self.params['bias'] - bias_old)
		# All the updates went through return that the update was successfull
		return 1

	def __choose_second_alpha(self, m, pos_alpha):
		if len(pos_alpha) > 0:
			# Case1: Choose the second alpha over non boundary examples with max error
			i = np.argmax(np.abs(self.Errors[m] - self.Errors[pos_alpha]))
			n = pos_alpha[i]
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
		start_idx = np.random.randint(num_train)
		for i in range(start_idx, num_train):
			took_step = self.__solver(m, i)
			if took_step:
				return 1

		# Casse 4: No significant change after looking at all other cases
		return 0

	def train(self, print_every=500):
		def print_obj_value():
			if (self.verbose) and (num_iter % print_every == 0):
				print "This is iteration {}".format(num_iter)
				print "The value of the objective function is: {}".format(obj_value)

		num_train = len(self.X)
		examine_all = 1
		num_changed = 0
		num_iter = 1
		while (num_changed > 0) or examine_all:
			num_changed = 0
			pos_alpha = [j for j in range(num_train) if (0 < self.params['alpha'][j]) and \
														(self.params['alpha'][j]) < self.C]
			# Loop through all the examples and pick the first alpha
			if examine_all:
				for i in range(num_train):
					choose_succeed = self.__choose_second_alpha(i, pos_alpha)
					num_changed += choose_succeed
					# Only add the obj value if a change was made
					if choose_succeed:
						obj_value = self.__evaluate_objective_function()
						self.objective_func_values.append(obj_value)
					print_obj_value()
					num_iter += 1

			else:
				for i in pos_alpha:
					choose_succeed = self.__choose_second_alpha(i, pos_alpha)
					num_changed += choose_succeed
					if choose_succeed:
						obj_value = self.__evaluate_objective_function()
						self.objective_func_values.append(obj_value)
					print_obj_value()
					num_iter += 1

			if examine_all == 1:
				examine_all = 0
			elif num_changed == 0:
				examine_all = 1

		# Plot the final model
		if self.verbose:
			print "(w:{}, b:{})".format(self.params['w'], self.params['bias'])
			fig, ax = plt.subplots()
			grid, ax = self.plot_solution(100, ax)
			plt.xlabel('petal_width')
			plt.ylabel('petal_length')
			plt.figure(figsize=(8,6))
			plt.show()

	def predict(self, X_test):
		temp = self.params['alpha'] * self.y
		y_pred = np.sum(self.__kernel(self.X, X_test) * temp[:,np.newaxis], axis=0) + self.params['bias']
		return y_pred

	def plot_solution(self, resolution, ax, colors=['b', 'k', 'r']):
		x_range = np.linspace(self.X[:,0].min(), self.X[:,0].max(), resolution)
		y_range = np.linspace(self.X[:,1].min(), self.X[:,1].max(), resolution)
		grid = [[self.predict(np.array([[xi, yi]]))[0] for xi in x_range] for yi in y_range]
		grid = np.array(grid)

		ax.contour(x_range, y_range, grid, levels=(-1,0,1), linewidths=(1,1,1),
                   linestyles=('--','-','--'), colors=colors)         
		ax.scatter(self.X[:,0], self.X[:,1], c=self.y, cmap=plt.cm.viridis, lw=0, alpha=0.5)
		mask = self.params['alpha'] != 0
		ax.scatter(self.X[:,0][mask], self.X[:,1][mask], c=self.y[mask], cmap=plt.cm.viridis, s=100)
		return grid, ax








