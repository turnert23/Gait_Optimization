import numpy as np
from scipy import stats
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import GPy



# Standard normal distribution functions
phi = stats.distributions.norm().pdf
PHI = stats.distributions.norm().cdf
PHIinv = stats.distributions.norm().ppf

# A few constants
lim = 8


def findGPUCB(y_pred,sigma, beta, xx):
  comp  = y_pred + beta*sigma;
  ind = comp.argmax()
  New = xx[ind]
  New = np.array([[New[0],New[1]]])#print g(New)
  return New


#This function returns a multi dimensional array, as need for GPY
def g1(x):
    """The function to predict (classification will then consist in predicting
    whether g(x) <= 0 or not)"""
    #return 64- x[:,1]**2*np.sin(x[:,0]) - x[:,0]**2*np.cos(x[:,1])
    l = np.size(X,0)
    return 64. - x[:, 1:2] ** 2. - .5 * x[:, 0:1] ** 2. + np.random.randn(l,1)
# This function returns a single dimensional array
def g(x):
    """The function to predict (classification will then consist in predicting
    whether g(x) <= 0 or not)"""
    return 64. - x[:, 1] ** 2. - .5 * x[:, 0] ** 2.


# Original Points
regret = []
X = np.array([[-4.61611719, -6.00099547],
              [4.10469096, 5.32782448],
              [-6.17289014, -4.6984743],
              [1.3109306, -6.93271427],
              [5.21301203, 4.26386883]])
#Number of iterations
T=15
for j in range(T):
  #calculate sample points
  y  = g1(X)
 
  #Create a kernel
  kernel = GPy.kern.RBF(input_dim=2, variance=20., lengthscale=1)

  #Create a model from the kernel
  m = GPy.models.GPRegression(X,y,kernel)
  
  #Create a discrete sample space (-lim to lim in x1 and x2
  #  with <res> sampled points in each direction)
  res = 50
  x1, x2 = np.meshgrid(np.linspace(- lim, lim, res),
                       np.linspace(- lim, lim, res))
  xx = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T

  #Calculate the true value of the function in the discrete space
  y_true = g(xx)

  # find new mean, sigma using the model's predict method
  mu, sig = m.predict(xx,full_cov=False,Y_metadata=None)
  
  #CREATE A DEEP COPY TO KEEP THE PLOT OPEN?
  #THIS MAY BE UNECCESARY
  y_pred = mu.copy()
  sigma = sig.copy()
  
  #Find next sample point
  New = findGPUCB(y_pred,sigma,1,xx)
  
  y_true = y_true.reshape((res, res))
  y_pred = y_pred.reshape((res, res))
  sigma = sigma.reshape((res, res))

  print New 
  
  #Plot intermediate mean, and covariance
  #THIS CANNOT BE DONE CURRENTLY FOR 
  # SOME REASON I DO NOT KNOW
  fig = pl.figure(1)
  #ax = fig.add_subplot(111, projection='3d')
  ax = pl.axes(projection='3d')

  c=x1+x2
  ax.scatter(X[:,0],X[:,1],y,s=30,c =u'g')
  ax.plot_wireframe(x1,x2,y_pred,colors=u'r')
  ax.plot_wireframe(x1,x2,y_pred+sigma)
  pl.show()
  
  #Add New point to list of samples
  X = np.append(X,New,0)
  #Calculate Euclidean Distance to function max
  #(This is done stupdily right now, change later)
  reg = New[0][0]**2+New[0][1]**2
  #append regret value to vector
  regret.append(reg)
#Plot regret 
#ALSO CANNOT BE DONE
fig  = pl.figure(1)
pl.plot(range(T),regret,'g-')
pl.show()
