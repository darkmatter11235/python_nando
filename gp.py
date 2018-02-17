import numpy as np
import matplotlib.pyplot as plt

def kernel(a, b, param):
    #(x-u)*(x-u).T
    #X**2 + u**2 - 2*np.dot(x,u.T)
    sqdist = np.sum(a**2,1).reshape(-1, 1) + np.sum(b**2, 1) - 2*np.dot(a, b.T)
    return np.exp(-0.5*(1.0/param)*sqdist)

#number of datapoints to fit
n = 50
n_samples = 10
epsilon = 0.00005
#draw these uniformly
Xtest = np.linspace(-5,5,n).reshape(-1,1)

#print Xtest.shape
#print np.sum(Xtest**2,1).reshape(-1,1)

params = [0.1, 1, 2]
fig = plt.figure()
index = 0
num_rows = len(params)
num_cols = 2
#define a training function
f  = lambda x: np.sin(0.9*x).flatten()
#define training Samples
n_train = 10
Xtrain = np.random.uniform(-5, 5, size=(n_train,1))
ytrain = f(Xtrain)
#Xtrain = np.linspace(-5, 5, n)

for param in params:

    K_ss = kernel(Xtest, Xtest, param)
    K_s = kernel(Xtrain, Xtest, param)
    K = kernel(Xtrain, Xtrain, param)
    #print Kss.shape
    #print K_ss
    #print param
    L = np.linalg.cholesky(K_ss + epsilon*np.eye(n))
    #print L.shape
    #print L
    r = np.random.normal(size=(n,n_samples))
    #print r.shape
    #print r
    #f_prior = np.dot(L, np.random.normal(size=(n,3)))
    f_prior = np.dot(L, r)
    #print f_prior.shape
    #print f_prior
    index += 1

    ax1 = fig.add_subplot('%d%d%d' %(num_rows, num_cols, index))

    ax1.plot(Xtest, f_prior)

    L = np.linalg.cholesky(K +  epsilon*np.eye(n_train))
    
    Lk = np.linalg.solve(L, K_s)
    #mu_s = mu_s + k_s.T*K_inv*y
    #
    mu_s = np.dot(Lk.T, np.linalg.solve(L, ytrain))

    variance_s = K_ss - np.dot(Lk.T, Lk)

    sigma_s = np.linalg.cholesky(variance_s + epsilon*np.eye(n))
    
    f_post = mu_s.reshape(-1,1) + np.dot(sigma_s, np.random.normal(size=(n, n_samples)))

    index += 1
    ax2 = fig.add_subplot('%d%d%d' %(num_rows, num_cols, index))
    ax2.plot(Xtrain, f(Xtrain), 'bo')
    ax2.plot(Xtest, f_post)
    #ax_train = fig.add_subplot('%d14' %(num_rows))
    #ax_train.plot(Xtrain, f(Xtrain))

plt.show()
