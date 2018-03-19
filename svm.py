import numpy



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

	def naive_max_utility(self, X, y):
		num_train  = len(X)
		self.params['alpha'] = np.zeros(num_train)


	def predict(self, X):
		y_pred = np.dot(X, self.params['w']) + self.params['bias']
		y_pred[y_pred >= 0] = 1
		y_pred[y_pred < 0] = -1
		return y_pred
