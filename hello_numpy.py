import numpy as np

A = np.matrix([[-2, 2, 3],[2, 1, -6],[-1, -2, 0]])

print A
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

print L

I = np.eye(3)

b = np.zeros(3).reshape(-1,1)

#
for l in np.nditer(L):
    print l
    LI = l*I
    print "LI"
    print LI
    x = np.linalg.solve(A-LI, b)
    print "x"
    print x


w, v = np.linalg.eig(A)

print w
print v
