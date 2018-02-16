import numpy as np
import matplotlib.pyplot as plt

def kernel(a, b, param):
    #(x-u)*(x-u).T
    #X**2 + u**2 - 2*np.dot(x,u.T)
    sqdist = np.sum(a**2,1).reshape(-1, 1) + np.sum(b**2, 1) - 2*np.dot(a, b.T)
    return np.exp(-0.5*(1.0/param)*sqdist)

#number of datapoints to fit
n = 50

#draw these uniformly
Xtest = np.linspace(-5,5,n).reshape(-1,1)

#print Xtest.shape
#print np.sum(Xtest**2,1).reshape(-1,1)

params = [0.1, 0.2, .3]
fig = plt.figure()
index = 0

for param in params:

    K_ss = kernel(Xtest, Xtest, param)

    #print Kss.shape
    #print K_ss

    print param

    L = np.linalg.cholesky(K_ss + 1e-15*np.eye(n))

    f_prior = np.dot(L, np.random.normal(size=(n,3)))

    index += 1

    ax1 = fig.add_subplot('31%d' %index)

    ax1.plot(Xtest, f_prior)

plt.show()
