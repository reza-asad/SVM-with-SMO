import numpy as np
import random
import matplotlib.pyplot as plt
from utility import *


class SVM():
	# Comments later
	def __init__(self, X, y, reg=0.0, dtype=np.float64, verbose=False):
		self.X = X
		self.y = y

		self.reg = reg
		self.dtype = dtype
		self.verbose = verbose
		self.params = {}

		# initialize the parameters of the model
		N, D = X.shape
		self.params['alpha'] = np.zeros(N)
		self.params['w'] = np.ones(D)
		self.params['bias'] = 1

	def __solver(self, m, n, idx, epsilon):
		# The utility function is a polynomial of degree 2 in alpha
		# a * alpha^2 + b * alpha + c
		# The solution is -b/(2*a)
		idx_subset = idx[:m] + idx[m+1:n] + idx[n+1:]
		X_subset = self.X[idx_subset,:]
		y_subset = self.y[idx_subset]
		alpha_subset = self.params['alpha'][idx_subset]
		x_m = self.X[m,:]
		y_m = self.y[m]
		x_n = self.X[n,:]
		y_n = self.y[n]

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
		# Clip the sollution to satisfy the optimizations constraint.
		if y_m * y_n == 1:
			gamma = self.params['alpha'][m] + self.params['alpha'][n]
			alpha_n = np.maximum(np.minimum(alpha_n, gamma), 0)
			alpha_m = np.maximum(np.minimum(alpha_m, gamma), 0)
		elif y_m * y_n == -1:
			gamma = self.params['alpha'][m] - self.params['alpha'][n]
			if gamma > 0:
				alpha_n = np.maximum(alpha_n, 0)
				alpha_m = np.maximum(alpha_m, gamma)
			else:
				alpha_n = np.maximum(alpha_n, -gamma)
				alpha_m = np.maximum(alpha_m, 0)

		# Update the pair of alpha		
		self.params['alpha'][m] = alpha_m
		self.params['alpha'][n] = alpha_n

		# Compute the weights given the new values of alpha
		self.params['w'] = np.sum(self.X * self.y[:,np.newaxis] * self.params['alpha'][:,np.newaxis], axis=0)

		# Compute the bias
		positive_X = self.X[(self.y==1),:]
		negative_X = self.X[(self.y==-1),:]
		self.params['bias'] = -0.5 * (np.max(np.dot(self.params['w'], negative_X.T)) + \
									  np.min(np.dot(self.params['w'], positive_X.T)))

	def __advanced_solver(self, m, n, threshold=1e-2):
		if m > n:
			m, n = n, m
		

	def naive_max_utility(self, epsilon=1e-5, print_every=1000):
		# The utility function is a polynomial of degree 2 in alpha
		# a * alpha^2 + b * alpha + c
		# The solution is -b/(2*a)
		num_train = len(self.X)
		idx = range(num_train)
		num_iter = 1
		while not kkt(self.X, self.y, self.params['alpha'], self.params['w'], self.params['bias']):
			m, n = random.sample(idx, 2)
			if m > n:
				m,n = n, m
			self.__solver(m, n, idx, epsilon)

			if self.verbose and (num_iter % print_every == 0):
				print "This is iteration {}:".format(num_iter)
				print "(w:{}, b:{})".format(self.params['w'], self.params['bias'])
				fig, ax = plt.subplots()
				grid, ax = self.plot_solution(200, ax)
				plt.show()
			num_iter += 1

		if self.verbose:
			print "Plot of the final model"
			fig, ax = plt.subplots()
			grid, ax = self.plot_solution(300, ax)
			plt.show()
		return self.params['alpha'], self.params['w'], self.params['bias']

	def max_utility(self, epsilon=1e-5, print_every=1000, success_threshold=100, pos_alpha_prob=0.6):
		num_train = len(self.X)
		num_success_kkt = 0
		num_iter = 1
		while (num_success_kkt < success_threshold):
			# Iterate through the non boundary alphas with probability pos_alpha_prob
			prob = np.random.uniform(0,1) 
			pos_alpha = [i for i in range(num_train) if self.params['alpha'][i] > 0]
			if (len(pos_alpha) > 0) and (prob < pos_alpha_prob:):
				m = np.random.randint(len(pos_alpha))
			else:
				m = np.random.randint(num_train)
			
			# Case 1: Choose alpha_n over the non-boundary examples that maximizes the E2 - E1
			if len(pos_alpha) > 0:
				E1_old = np.abs(self.predict(np.array([self.X[m,:]])) - self.y[m])
				E2_old = E1_old
				for i in pos_alpha:
					E2_candidate = np.abs(self.predict(np.array([self.X[i,:]])) - self.y[i])
					if abs(E2_candidate - E1_old) > abs(E2_old - E1_old):
						E2_old = E2_candidate
						n = i

				update_successful  = self.__advanced_solver(m,n)
				if not update_successful:
					random_start_idx = np.random.randint(len(pos_alpha))
					for i in range(random_start_idx, len(pos_alpha)):
						update_successful = self.__advanced_solver(m,i):
						if update_successful:
							break

			if not update_successful:
				random_start_idx = np.random.randint(num_train)
				for i in range(random_start_idx, num_train):
					if self.__advanced_solver(m, i):
						break
			# Check if the new solution violates kkt
			if kkt(self.X, self.y, self.params['alpha'], self.params['w'], self.params['bias']):
				num_success_kkt += 1
			else:
				num_success_kkt = 0

			# Plot the evolution of the solution
			if self.verbose and (num_iter % print_every == 0):
				print "This is iteration {}:".format(num_iter)
				print "(w:{}, b:{})".format(self.params['w'], self.params['bias'])
				fig, ax = plt.subplots()
				grid, ax = self.plot_solution(200, ax)
				plt.show()
			num_iter += 1

		# Plot the final model
		if self.verbose:
			print "Plot of the final model"
			fig, ax = plt.subplots()
			grid, ax = self.plot_solution(300, ax)
			plt.show()

		return self.params['alpha'], self.params['w'], self.params['bias']

	def predict(self, X_test, epsilon=0):
		y_pred = np.dot(X_test, self.params['w']) + self.params['bias']
		y_pred[(y_pred-epsilon) >= 0] = 1
		y_pred[(y_pred+epsilon) < 0] = -1
		return y_pred

	def plot_solution(self, resolution, ax, colors=['b', 'k', 'r']):
		x_range = np.linspace(self.X[:,0].min(), self.X[:,0].max(), resolution)
		y_range = np.linspace(self.X[:,1].min(), self.X[:,1].max(), resolution)
		grid = [[self.predict(np.array([[xi, yi]]), epsilon=1.01)[0] for xi in x_range] for yi in y_range]
		grid = np.array(grid)

		ax.contour(x_range, y_range, grid, levels=(-1,0,1), linewidths=(1,1,1),
                   linestyles=('--','-','--'), colors=colors)         
		ax.scatter(self.X[:,0], self.X[:,1], c=self.y, cmap=plt.cm.viridis, lw=0, alpha=0.5)
		mask = self.params['alpha'] > 0
		ax.scatter(self.X[:,0][mask], self.X[:,1][mask], c=self.y[mask], cmap=plt.cm.viridis, s=100)
		return grid, ax









