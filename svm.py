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


	def __solver(self, m, n, epsilon=1e-5, update_threshold=1e-3):
		# The utility function is a polynomial of degree 2 in alpha
		# a * alpha^2 + b * alpha + c
		# The solution is -b/(2*a)
		if m > n:
			m ,n = n, m
		idx = range(len(self.X))
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

		# Em_old = (self.predict(np.array([self.X[m,:]])) - self.y[m])[0]
		# En_old = (self.predict(np.array([self.X[n,:]])) - self.y[n])[0]
		# k = -np.dot(self.X[m,:], self.X[m,:]) - np.dot(self.X[n,:], self.X[n,:]) + \
		# 	2 * np.dot(self.X[n,:], self.X[m,:])
		# alpha_n_prime = self.params['alpha'][n] + self.y[n] * (En_old - Em_old) / (k+1e-5)
		# alpha_m_prime = self.params['alpha'][m] + self.y[m] * self.y[n] * (self.params['alpha'][n] - alpha_n_prime)

		# Clip the sollution to satisfy the optimizations constraint.
		alpha_m, alpha_n = self.__clip(m, n, alpha_m, alpha_n)
		# alpha_m_prime, alpha_n_prime = self.__clip(m, n, alpha_m_prime, alpha_n_prime)

		# If there is not enough change in the parameter skip the update
		alpha_m_static = abs(alpha_m - self.params['alpha'][m]) < update_threshold
		alpha_n_static = abs(alpha_n - self.params['alpha'][n]) < update_threshold
		if alpha_m_static or alpha_n_static:
			return 0

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
		return 1

	def __advanced_solver(self, m, n, epsilon, update_threshold=1e-5):
		if m > n:
			m, n = n, m
		Em_old = self.predict(np.array([self.X[m,:]])) - self.y[m]
		En_old = self.predict(np.array([self.X[n,:]])) - self.y[n]
		k = -np.dot(self.X[m,:], self.X[m,:]) - np.dot(self.X[n,:], self.X[n,:]) + \
			2 * np.dot(self.X[n,:], self.X[m,:])
		alpha_n = self.params['alpha'][n] + self.y[n] * (En_old - Em_old) / (k+epsilon)
		alpha_m = self.params['alpha'][m] + self.y[m] * self.y[n] * (self.params['alpha'][n] - alpha_n)
		alpha_m, alpha_n = self.__clip(m, n, alpha_m, alpha_n)

		# If there is not enough change in the parameter skip the update
		alpha_m_static = abs(alpha_m - self.params['alpha'][m]) < update_threshold
		alpha_n_static = abs(alpha_n - self.params['alpha'][n]) < update_threshold
		if alpha_m_static and alpha_n_static:
			return False

		# Update alphas
		self.params['alpha'][m] = alpha_m
		self.params['alpha'][n] = alpha_n
		# Compute the weights given the new values of alpha
		self.params['w'] = np.sum(self.X * self.y[:,np.newaxis] * self.params['alpha'][:,np.newaxis], axis=0)
		# Compute the bias
		positive_X = self.X[(self.y==1),:]
		negative_X = self.X[(self.y==-1),:]
		self.params['bias'] = -0.5 * (np.max(np.dot(self.params['w'], negative_X.T)) + \
									  np.min(np.dot(self.params['w'], positive_X.T)))
		return True


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

	def train(self, print_every=1000):
		num_train = len(self.X)
		examine_all = 1
		num_changed = 0
		num_iter = 1
		while (num_changed > 0) or examine_all:
			num_changed = 0
			pos_alpha = [j for j in range(num_train) if self.params['alpha'][j] > 0]
			# Loop through all the examples and pick the first alpha
			if examine_all:
				for i in range(num_train):
					choose_succeed = self.__choose_second_alpha(i, pos_alpha)
					num_changed += choose_succeed
					num_iter += 1
					# Plot the evolution of the solution
					if self.verbose and (num_iter % print_every == 0):
						print "This is iteration {}:".format(num_iter)
						print "(w:{}, b:{})".format(self.params['w'], self.params['bias'])
						fig, ax = plt.subplots()
						grid, ax = self.plot_solution(200, ax)
						plt.show()

			else:
				for i in pos_alpha:
					choose_succeed = self.__choose_second_alpha(i, pos_alpha)
					num_changed += choose_succeed
					num_iter += 1
					# Plot the evolution of the solution
					if self.verbose and (num_iter % print_every == 0):
						print "This is iteration {}:".format(num_iter)
						print "(w:{}, b:{})".format(self.params['w'], self.params['bias'])
						fig, ax = plt.subplots()
						grid, ax = self.plot_solution(200, ax)
						plt.show()

			if examine_all == 1:
				examine_all = 0
			elif num_changed == 0:
				examine_all = 1

			# Plot the evolution of the solution
			if self.verbose and (num_iter % print_every == 0):
				print "This is iteration {}:".format(num_iter)
				print "(w:{}, b:{})".format(self.params['w'], self.params['bias'])
				fig, ax = plt.subplots()
				grid, ax = self.plot_solution(200, ax)
				plt.show()


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









