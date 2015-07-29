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
		"""default 'true' function. Returns a noisy sample of the value 
		self.f(xx). For applications, simply override this method to link to
		an outside function evaluation."""

		yy = self.f(xx) + randn()

		return yy

	def addRandomPoint(self):
		"""Picks a randomly-selected point xx = randn(), then calls 
		samplePoint(xx) and adds the resulting data to the object."""

		xx = randn()

		yy = self.samplePoint(xx)

		self.updateData(xx,yy)

	def updateData(self, xx, yy):
		"""Adds a data point (xx, yy) to the data structure."""

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
		"""Builds a GPRegression model using X, Y, and the kernel, then 
		optimizes the parameter values."""

		self.m = GPy.models.GPRegression(self.X, self.Y, self.kernel)

		self.m.optimize()

	def UCB(self, xx, alpha):
		"""Computes the (1-alpha)th quantile of the posterior distribution
		at the location xx"""

		# Ensure that data is of appropriate array dimension
		xx = array(xx, ndmin=2)

		mean, var = self.m.predict(xx)

		Q = mean - sqrt(var)*stats.norm.ppf(alpha)

		return Q

	def maxUCB(self, alpha):
		"""Finds the value of xx that maximizes the (1-alpha)th quantile of 
		the posterior distribution. Uses scipy.optimize.minimize."""

		