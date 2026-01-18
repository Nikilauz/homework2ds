import numpy as np
import matplotlib.pyplot as plt

from gradient_descent import argmin_F

autodot = lambda x : np.inner(x, x)
np.set_printoptions(precision=2)

N = 10000               # dimension
STEPOUT = 5000          # max iterations
RAND_SIZE = 2**10       # range of randomly drawn numbers
GUESS_EFFORT = 100      # how much effort to guess beta 0
TOLERANCE = 1e-3        # convergence criterium

hyperpars = {
    "N": N,
    "STEPOUT": STEPOUT,
    "RAND_SIZE": RAND_SIZE,
    "GUESS_EFFORT": GUESS_EFFORT,
    "TOLERANCE": TOLERANCE
}

############## Lasso recovery ######################

# set up Lasso problem
m = np.random.randint(low=N, high=1.5*N)
print("dimension m: " + str(m))
A = np.random.normal(size=(m, N))
w = np.random.uniform(low=-RAND_SIZE, high=RAND_SIZE, size=N)
sigma = 0.1
xi = np.random.normal(size=m)

y = (A @ w) + sigma * xi

# oracles
f = lambda x : 0.5 * autodot((A @ x) - y)
df = lambda x : np.transpose(A) @ ((A @ x) - y)

lamb = 2 * sigma * np.sqrt(np.log(N))

# launch gradient descent
w_recovered = argmin_F(f, df, lamb, hyperpars)
#print(w)
#print(w_recovered)
print(np.linalg.norm(w - w_recovered, 1))

#plt.show()
