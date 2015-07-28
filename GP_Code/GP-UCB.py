import GPy
from scipy import stats

class GPOpt(object):
	"""docstring for GPOpt class"""
	def __init__(self):
		# Set up data structures for the input and output data
		self.X = array([[]])
		self.Y = array([[]])

		self.X.shape = (0, 1)
		self.Y.shape = (0, 1)

		self.kernel = GPy.kern.RBF(input_dim=1, variance=1., lengthscale=1.)

		self.f = lambda x: cumprod(sin(xx))[-1]

	def samplePoint(self, xx):
		"""default 'true' function"""

		yy = self.f(xx) + randn()

		return yy

	def addRandomPoint(self):
		xx = randn()

		yy = self.samplePoint(xx)

		self.updateData(xx,yy)

	def updateData(self, xx, yy):
		# Add a new data point to the data structure
		# First, check that the data are of appropriate dimension
		xx = array(xx, ndmin=2)
		yy = array(yy, ndmin=2)

		# Then, append
		self.X = np.append(self.X, xx, axis=0)
		self.Y = np.append(self.Y, yy, axis=0)

		# Verify that arrays are the right size
		self.X = array(self.X, ndmin=2)
		self.Y = array(self.Y, ndmin=2)

	def optimizeModel(self):
		self.m = GPy.models.GPRegression(self.X, self.Y, self.kernel)

		self.m.optimize()

	def UCB(self, xx, alpha):
		mean, var = self.m.predict(xx)

		Q = mean - sqrt(var)*stats.norm.ppf(alpha)

		return Q