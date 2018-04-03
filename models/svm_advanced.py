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
		pass

	def train(self, print_every=500):
		super(SVMAdvanced, self).train(check_linearity=False, print_every=print_every)









