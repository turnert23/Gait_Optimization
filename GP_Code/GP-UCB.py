import GPy
from scipy import stats
from scipy.optimize import minimize
from scipy.optimize import minimize_scalar

class GPOpt(object):
	"""docstring for GPOpt class"""
	def __init__(self, dim=1):
		# Set up data structures for the input and output data
		self.X = array([[]])
		self.Y = array([[]])

		# Dimension of the process (number of variables)
		self.dim = dim

		self.X.shape = (0, dim)
		self.Y.shape = (0, 1)

		self.kernel = GPy.kern.RBF(input_dim=dim, variance=1., lengthscale=1.)

		self.f   = lambda x: cumprod(sin(x))[-1]
		self.std = 0.5

		#self.fig     = plt.figure()
		#self.ax  	  = plt.plot()
		#self.ax.hold = False

		self.bounds = (-pi, pi)
		if dim == 2:
			self.bounds = ((-pi, pi), (-pi, pi))
		# for self.dim > 1, format is ((-pi, pi), (-pi, pi), ...)

		self.bounds = [(-pi, pi) for i in range(dim)]

	def samplePoint(self, xx):
		"""default 'true' function. Returns a noisy sample of the value 
		self.f(xx). For applications, simply override this method to link to
		an outside function evaluation."""

		yy = self.f(xx) + self.std*randn()

		return yy

	def addRandomPoint(self):
		"""Picks a randomly-selected point xx = randn(), then calls 
		samplePoint(xx) and adds the resulting data to the object."""
		
		if self.dim == 1:
			xx = randn()
		else:
			xx = randn(self.dim)

		yy = self.samplePoint(xx)

		self.addData(xx,yy)

	def addData(self, xx, yy):
		"""Adds a data point (xx, yy) to the data structure."""

		# Add a new data point to the data structure
		# First, check that the data are of appropriate dimension
		xx = atleast_2d(xx)
		yy = atleast_2d(yy)

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

		# Define the optimization objective (with negative sign so we maximize)
		func = lambda x: -self.UCB(x, alpha)

		if self.dim == 1:
			# Initial condition
			x0 = randn()

			# Call the optimizer (may need to add constraints later)
			res = minimize_scalar(func, bounds=self.bounds, method='Bounded')
			
		else:		# multi-dimensional
			x0 = randn(self.dim)

			res = minimize(func, x0, bounds=self.bounds)

		xxMax = res.x

		yyMax = -res.fun

		return xxMax, yyMax

	def initialize(self, n=10):
		'''Code to take samples on a coarse grid to initialize the regression'''
		if self.dim == 1:
			xxInit = linspace(self.bounds[0], self.bounds[1], n)
		elif self.dim == 2:
			xx0 = linspace(self.bounds[0][0], self.bounds[0][1], n)
			xx1 = linspace(self.bounds[1][0], self.bounds[1][1], n)

			xx, yy = meshgrid(xx0, xx1)

			xxInit = transpose(array((ravel(xx), ravel(yy))))

		else:
			xx = meshgrid(*[np.linspace(i, j, n) for i,j in self.bounds])

			xxInit = array([ravel(xx[i]) for i in range(self.dim)]).transpose()


		for xx in xxInit:
			self.sampleAndUpdate(xx)

	def plotUCB(self, alpha=0.33, n=50):
		'''Code is only written for dim=1; can be rewritten for dim > 1'''
		xx = linspace(self.bounds[0], self.bounds[1], n)
		yy = zeros_like(xx)

		for ii in arange(n):
			yy[ii] = self.UCB(xx[ii], alpha)

		plot(xx, yy)


	def gpUCB(self, T=100, n=10):
		
		# Generate two data points for initialization
		#self.addRandomPoint()
		#self.addRandomPoint()

		# Initialize the model
		#self.optimizeModel()
		self.initialize(n)

		if self.dim <= 2:
			fig, self.ax = plt.subplots()
			self.ax.hold = False

			self.m.plot(ax = self.ax)

		# Run main loop
		for tt in range(2+n, T+n):
			alpha = 1./tt
			xxNew, yyNew = self.maxUCB(alpha)

			self.sampleAndUpdate(xxNew)

			if self.dim <= 2:
				self.m.plot(ax = self.ax)

	def runGPUCB(self, T=100, n=10):
		self.gpUCB(T, n)

		self.mit = zeros((T+n**self.dim - 2, 1))
		self.rt  = zeros((T+n**self.dim - 2, 1))

		for tt in range(T+n**self.dim-2):
			self.mit[tt] = self.f(self.X[tt, :])
			self.rt[tt]  = 1 - self.mit[tt]

		if self.dim <= 2:
			figure()

			self.m.plot()

		figure()

		if self.dim == 1:	# Need to fix for dim>1: make a contour plot
			self.plotUCB(alpha=0.5)

		figure()

		plot(cumsum(self.rt))

	def runEnsemble(self, T=100, n=10, nEnsemble=100):
		self.rts = zeros((T+n, nEnsemble))

		for run in range(nEnsemble):
			self.gpUCB(T, n)

			for tt in range(T+n-2):
				mit = self.f(self.X[tt, :])

				self.rts[tt, run] = 1 - mit
