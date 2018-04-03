import numpy as np
import random
import matplotlib.pyplot as plt

from svm_basic import *


class SVMAdvanced(SVM):

	def train(self, print_every=500):
		super(SVMAdvanced, self).train(check_linearity=False, print_every=print_every)









