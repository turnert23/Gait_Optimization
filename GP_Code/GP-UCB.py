import GPy
from scipy import stats
from scipy.optimize import minimize

class GPOpt(object):
	"""docstring for GPOpt class"""
	def __init__(self, dim=1):
		# Set up data structures for the input and output data
		self.X = array([[]])
		self.Y = array([[]])

		# Dimension of the process (number of variables)
		self.dim = dim

		self.X.shape = (0, dim)
		self.Y.shape = (0, dim)

		self.kernel = GPy.kern.RBF(input_dim=dim, variance=1., lengthscale=1.)

		self.f   = lambda x: cumprod(sin(x))[-1]
		self.std = 0.5

		#self.fig     = plt.figure()
		#self.ax  	  = plt.plot()
		#self.ax.hold = False

	def samplePoint(self, xx):
		"""default 'true' function. Returns a noisy sample of the value 
		self.f(xx). For applications, simply override this method to link to
		an outside function evaluation."""

		yy = self.f(xx) + self.std*randn()

		return yy

	def addRandomPoint(self):
		"""Picks a randomly-selected point xx = randn(), then calls 
		samplePoint(xx) and adds the resulting data to the object."""

		xx = randn()

		yy = self.samplePoint(xx)

		self.addData(xx,yy)

	def addData(self, xx, yy):
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

	def sampleAndUpdate(self, xx):
		"""Samples the point xx, updates the data structure and model"""

		yy = self.samplePoint(xx)

		self.addData(xx, yy)

		self.optimizeModel()

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

		# Initial condition
		x0 = randn()

		# Define the optimization objective (with negative sign so we maximize)
		func = lambda x: -self.UCB(x, alpha)

		# Call the optimizer (may need to add constraints later)
		res = minimize(func, x0)

		xxMax = res.x

		yyMax = -res.fun

		return xxMax, yyMax

	def gpUCB(self, T=100):
		
		# Generate two data points for initialization
		self.addRandomPoint()
		self.addRandomPoint()

		# Initialize the model
		self.optimizeModel()

		fig, self.ax = plt.subplots()
		self.ax.hold = False

		self.m.plot(ax = self.ax)

		# Run main loop
		for tt in range(2, T+1):
			alpha = 1./tt
			xxNew, yyNew = self.maxUCB(alpha)

			self.sampleAndUpdate(xxNew)

			self.m.plot(ax = self.ax)

		