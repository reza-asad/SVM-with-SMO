import numpy as np
import random
import matplotlib.pyplot as plt

from svm_basic import *


class SVMAdvanced(SVM):
	def __clip(self, m ,n, alpha_m, alpha_n):
		pass

	def __solver(self):
		pass

	def __choose_second_alpha(m, pos_alpha):
		if len(pos_alpha) > 0:
			# Case1: Choose the second alpha over non boundary examples with max error
			n = np.argmax(self.Errors - self.Errors)
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
		start_idx = np.random.randint(start_idx, num_train):
		for i in range(start_idx, num_train):
			took_step = self.__solver(m, i)
			if took_step:
				return 1

		# Casse 4: No significant change after looking at all other cases
		return 0

	def train(self, print_every=500):
		super(SVMAdvanced, self).train(check_linearity=False, print_every=print_every)









