import numpy as np
import matplotlib.pyplot as plt
import math

A = np.matrix([[-2, 2, 3],[2, 1, -6],[-1, -2, 0]])

#print A
#(A*e = l*e)
#-2 2  3
# 2 1 -6
#-1 -2 0

# -2-l  2   3
#  2   1-l -6
# -1  -2   -l

# (-2-l)*(-l+l**2-12) - 2(-2l-6) + 3 ( -4 + 1 -l )
# 2l - 2 l**2 + 24 + l**2 - l**3 + 12l + 4l + 12 - 12 + 3 - 3l
# -l**3 - 1 l**2 + 15l + 27

coeffs = [-1, -1, 15, 27]
L = np.roots(coeffs)

#print L

I = np.eye(3)

b = np.zeros(3)

#epsilon = 1e-16
#b = epsilon*b
# TODO get non-zero solutions
for l in np.nditer(L):
    #print l
    LI = l*I
    #print "A"
    #print A-LI
    #print "B"
    #print b
    x = np.linalg.tensorsolve(A-LI, b)
    #print "X"
    #print x


#w, v = np.linalg.eig(A)
#print w
#print v

#gaussian PDF
def gaussianPDF(mu, sigma, x):
    num = -((x-mu)**2)
    denom = 2*sigma**2
    return (1/(math.sqrt(2. * math.pi)*sigma))*(math.exp(num/denom))

mu = 0
sigma = 2
samples = np.arange(-5.0, 5, 0.05)
plt.xlabel('PDF')
for sigma in np.arange(0.5, 3, 0.01):
    plt.plot(samples, [gaussianPDF(mu, sigma, x) for x in samples ], label=sigma)
plt.legend()
plt.show()
