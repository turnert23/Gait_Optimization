import numpy as np
from scipy import stats
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import random

# Standard normal distribution functions
phi = stats.distributions.norm().pdf
PHI = stats.distributions.norm().cdf
PHIinv = stats.distributions.norm().ppf

# A few constants
lim = 8

# function to determine which new point to sample
# beta is weight of covariance, 
# xx is unrolled, discretized sample space
def findGPUCB(y_pred,sigma, beta, xx):
  comp  = y_pred + beta*sigma;
  ind = comp.argmax()
  New = xx[ind]
  New = np.array([[New[0],New[1]]])#print g(New)
  return New



def g(x):
    """The function to predict (classification will then consist in predicting
    whether g(x) <= 0 or not)"""
    #return 64- x[:,1]**2*np.sin(x[:,0]) - x[:,0]**2*np.cos(x[:,1])
    #return (np.sin(x[:, 1]) ** 2. + .5 * np.sin(x[:, 0]) ** 2.)/(x[:,1]+x[:,0])
    return 64. - x[:,1]**2 - 5.* x[:,0] ** 2


# Original Points
regret = []
X = np.array([[-4.61611719, -6.00099547],
              [4.10469096, 5.32782448],
              [-6.17289014, -4.6984743],
              [1.3109306, -6.93271427],
              [5.21301203, 4.26386883]])
#Number of iterations
T = 20

for j in range(T): 
  
  #Observations and noise
  y = g(X)
  dy = 0.5 + 1.0 * np.random.random(y.shape)
  noise = np.random.normal(0, dy)
  y += noise
  
  # Instanciate and fit Gaussian Process Model
  gp = GaussianProcess(corr='squared_exponential', theta0=1e-1,
                     thetaL=1e-3, thetaU=1,
                     nugget=(dy / y) ** 2,
                     random_start=100)

  #Update Gaussian Process
  gp.fit(X, y)

  # Evaluate real function, the prediction and its MSE on a grid
  res = 50
  x1, x2 = np.meshgrid(np.linspace(- lim, lim, res),
                       np.linspace(- lim, lim, res))
  xx = np.vstack([x1.reshape(x1.size), x2.reshape(x2.size)]).T

  y_true = g(xx)
  
  # find new mean, sigma
  y_pred, MSE = gp.predict(xx, eval_MSE=True)
  sigma = np.sqrt(MSE)
  y_true = y_true.reshape((res, res))
  y_pred = y_pred.reshape((res, res))
  sigma = sigma.reshape((res, res))
  
  # Find new maximum (mu + beta*sigma) to sample
  New = findGPUCB(y_pred,sigma,1.,xx)
 
  '''Uncomment lines 83-91 to plot each individual
  mean and cov. update.'''

  '''fig = pl.figure(1)
  ax = fig.add_subplot(111, projection='3d')
  ax = pl.axes(projection='3d')

  c=x1+x2
  ax.scatter(X[:,0],X[:,1],y,s=30,c =u'g')
  ax.plot_wireframe(x1,x2,y_pred,colors=u'r')
  ax.plot_wireframe(x1,x2,y_pred+sigma)
  pl.show()'''
  print New
  #Calculate the Euclidean distance of
  # new point to the maximum
  reg = New[0][0]**2+New[0][1]**2
  
  #append new regret value to regret vector
  regret.append(reg)
  
  #scikit's GP can't handle multiple data points (X)
  # with different output values (y_pred), so 
  # when converging, add a little random value
  # to X so that no two X's will be exactly the same 
  if New in X:
    New[0][0]+=random.random()/10
    New[0][1]+=random.random()/10
  X = np.append(X,New,0)

#plot regret
fig = pl.figure()
pl.plot(range(T),regret,'g-')
pl.show()

