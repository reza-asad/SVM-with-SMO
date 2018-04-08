import numpy as np
import random
import matplotlib.pyplot as plt


class SVM(object):
	# Comments later
	def __init__(self, X, y, dtype=np.float64, verbose=False):
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

	def evaluate_objective_function(self):
		temp = self.y * self.params['alpha']
		result = np.sum(self.params['alpha']) - 0.5 * np.dot(np.dot(temp, 
														    		np.dot(self.X,
														   		    	   self.X.T)
														    		),
															temp)
		return result

	def __is_data_linearly_seperable(self):
		_, D = self.X.shape
		for d in range(D):
			min_val = self.X[self.y==1,d:d+1].min()
			max_val = self.X[self.y==1,d:d+1].max()

			# This checks if in dimension d the data for one class is
			# within the range of data for the other class. If thats te
			# case we have a non linearly seperable dataset.
			I = (min_val < self.X[self.y == -1, d:d+1]) & \
				(self.X[self.y == -1, d:d+1] < max_val)
			return sum(I) == 0


	def __clip(self,m, n, alpha_m, alpha_n):
		if self.y[m] * self.y[n] == 1:
			gamma = self.params['alpha'][m] + self.params['alpha'][n]
			alpha_n = np.maximum(np.minimum(alpha_n, gamma), 0)
			alpha_m = np.maximum(np.minimum(alpha_m, gamma), 0)
		elif self.y[m] * self.y[n] == -1:
			gamma = self.params['alpha'][m] - self.params['alpha'][n]
			if gamma > 0:
				alpha_n = np.maximum(alpha_n, 0)
				alpha_m = np.maximum(alpha_m, gamma)
			else:
				alpha_n = np.maximum(alpha_n, -gamma)
				alpha_m = np.maximum(alpha_m, 0)
		return alpha_m, alpha_n

	def __solver(self, m, n, epsilon=1e-7):
		# Skip the case that both alphas are the same.
		if m == n:
			return 0

		if m > n:
			m, n = n, m

		# Our old parameters
		alpha_n_old = self.params['alpha'][n]
		alpha_m_old = self.params['alpha'][m]
		bias_old = self.params['bias']
		w_old = self.params['w']

		# The error based on our old parameters
		Em = np.dot(self.X[m,:], self.params['w']) + self.params['bias'] - self.y[m]
		En = np.dot(self.X[n,:], self.params['w']) + self.params['bias'] - self.y[n]

		k = -np.dot(self.X[m,:], self.X[m,:]) - np.dot(self.X[n,:], self.X[n,:]) + \
			2 * np.dot(self.X[n,:], self.X[m,:])

		# Computing the new alphas
		alpha_n = alpha_n_old + self.y[n] * (En - Em) / (k+epsilon)
		alpha_m = alpha_m_old + self.y[m] * self.y[n] * (alpha_n_old - alpha_n)

		# Clip the solution to make sure that it satisfies the constraint.
		alpha_m, alpha_n = self.__clip(m, n, alpha_m, alpha_n)

		# If there is not enough change in the parameter skip the update
		if abs(alpha_n - alpha_n_old) < (alpha_n + alpha_n_old + epsilon) * epsilon:
			return 0

		# Update alphas
		self.params['alpha'][m] = alpha_m
		self.params['alpha'][n] = alpha_n
		
		# Compute the weights given the new values of alpha
		self.params['w'] = w_old + (alpha_n - alpha_n_old) * self.y[n] * self.X[n,:] + \
								   (alpha_m - alpha_m_old) * self.y[m] * self.X[m,:]
		
		# Compute the bias
		# Take the bias that enforces the error to be zero for a non-boundary example (support vector).
		# If such bias can't be found take the average of the bias corresponding to each alpha.
		bias_m_new = bias_old - Em - (alpha_m - alpha_m_old) * self.y[m] * np.dot(self.X[m,:], self.X[m,:]) - \
									 (alpha_n - alpha_n_old) * self.y[n] * np.dot(self.X[m,:], self.X[n,:])

		bias_n_new = bias_old - En - (alpha_n - alpha_n_old) * self.y[n] * np.dot(self.X[n,:], self.X[n,:]) - \
					   			 	 (alpha_m - alpha_m_old) * self.y[m] * np.dot(self.X[m,:], self.X[n,:])
		if alpha_n > 0:
			self.params['bias'] = bias_n_new
		elif alpha_m > 0:
			self.params['bias'] = bias_m_new
		else:
			self.params['bias'] = 0.5 * (bias_n_new + bias_m_new)

		return 1


	def __choose_second_alpha(self, m, pos_alpha):
		if len(pos_alpha) > 0:
			# Case1: Choose the second alpha over non boundary examples with max error
			Em = np.abs(np.dot(self.X[m,:], self.params['w']) + \
				 self.params['bias'] - self.y[m])
			En = Em
			n = 0
			for i in pos_alpha:
				E_candidate = np.abs(np.dot(self.X[i,:], self.params['w']) + \
							  self.params['bias'] - self.y[i])
				if abs(Em - E_candidate) > abs(Em - En):
					En = E_candidate
					n = i
			took_step = self.__solver(m, n)
			if took_step:
				return 1

			# Case2: Choose the second alpha over the all non boundary examples 
			#		 starting from a random index.
			random_start_idx = np.random.randint(len(pos_alpha))
			for i in pos_alpha[random_start_idx:]:
				took_step = self.__solver(m, i)
				if took_step:
					return 1

		# Case3: Choose the second alpha over all the examples starting from a 
		#		 random index.
		num_train = len(self.X)
		random_start_idx = np.random.randint(num_train)
		for i in range(random_start_idx, num_train):
			took_step = self.__solver(m, i)
			if took_step:
				return 1
		return 0

	def train(self, print_every=500, check_linearity=True, alpha_lower_bound=0, 
			  alpha_upper_bound=float('inf')):
		if check_linearity:
			if not self.__is_data_linearly_seperable():
				print "Data is not linearly seperable. Please use the svm_advanced.py"
				return 
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
			pos_alpha = [j for j in range(num_train) if (alpha_lower_bound < self.params['alpha'][j]) and \
														(alpha_upper_bound > self.params['alpha'][j])]
			# Loop through all the examples and pick the first alpha
			if examine_all:
				for i in range(num_train):
					choose_succeed = self.__choose_second_alpha(i, pos_alpha)
					num_changed += choose_succeed
					# Only add the obj value if a change was made
					if choose_succeed:
						obj_value = self.evaluate_objective_function()
						self.objective_func_values.append(obj_value)
					print_obj_value()
					num_iter += 1

			else:
				for i in pos_alpha:
					choose_succeed = self.__choose_second_alpha(i, pos_alpha)
					num_changed += choose_succeed
					if choose_succeed:
						obj_value = self.evaluate_objective_function()
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
			grid, ax = self.plot_solution(200, ax)
			plt.xlabel('petal_width')
			plt.ylabel('petal_length')
			plt.show()


	def predict(self, X_test):
		y_pred = np.dot(X_test, self.params['w']) + self.params['bias']
		return y_pred

	def plot_solution(self, resolution, ax, colors=['b', 'k', 'r']):
		x_range = np.linspace(self.X[:,0].min(), self.X[:,0].max(), resolution)
		y_range = np.linspace(self.X[:,1].min(), self.X[:,1].max(), resolution)
		grid = [[self.predict(np.array([xi, yi])) for xi in x_range] for yi in y_range]
		grid = np.array(grid)

		ax.contour(x_range, y_range, grid, levels=(-1,0,1), linewidths=(1,1,1),
                   linestyles=('--','-','--'), colors=colors)         
		ax.scatter(self.X[:,0], self.X[:,1], c=self.y, cmap=plt.cm.viridis, lw=0, alpha=0.5)
		mask = self.params['alpha'] > 0
		ax.scatter(self.X[:,0][mask], self.X[:,1][mask], c=self.y[mask], cmap=plt.cm.viridis, s=100)
		return grid, ax









