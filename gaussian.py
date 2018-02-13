import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#gaussian PDF
def gaussianPDF(mu, sigma, x):
    num = -((x-mu)**2)
    denom = 2*sigma**2
    return (1/(math.sqrt(2. * math.pi)*sigma))*(math.exp(num/denom))

mu = 0
sigma = 2
samples = np.arange(-5.0, 5, 0.05)
fig = plt.figure()
ax1 = fig.add_subplot(131)
plt.xlabel('PDF')
for sigma in np.arange(0.5, 2, 0.1):
    ax1.plot(samples, [gaussianPDF(mu, sigma, x) for x in samples ], label=sigma)
ax1.legend()
#plt.show()

def gaussianPDF_2D(mu, sigma, x):
    #vectorized implmenetation
    delta = x-mu
    sig_inv = np.linalg.inv(sigma)
    delta_T = delta.copy()
    delta_T = delta_T.reshape(1,-1)
    delta = delta.reshape(-1,1)
    #print delta_T.shape
    #print sig_inv.shape
    #print delta.shape
    uhe =  (delta_T*sig_inv*delta)*-0.5
    lh = 2*math.pi*np.linalg.det(sigma)**0.5
    pdf = np.exp(uhe)/lh
    return float(pdf)


Sigma = np.matrix([[1.0, 0.], [0., 1.0]])
mu = np.array([0., 0.])
x = np.arange(-5, 5, 0.05)
y = np.arange(-5, 5, 0.05)
X, Y = np.meshgrid(x, y)

#print X.shape
nx, ny = X.shape
Z = np.zeros(X.shape)
for i in xrange(nx):
    for j in xrange(ny):
        Z[i,j] = gaussianPDF_2D(mu, Sigma, np.array([X[i,j], Y[i,j]]))

ax2 = fig.add_subplot(132)
plt.xlabel('Contour')
ax2.contour(X, Y, Z)
ax2.legend()

ax3 = fig.add_subplot(133, projection='3d')
#ax = Axes3D(ax3)
plt.xlabel('Surface')
ax3.legend()
ax3.plot_surface(X, Y, Z)
plt.show()
