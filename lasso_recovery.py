import numpy as np
import matplotlib.pyplot as plt

from gradient_descent import argmin_F, autodot

def prox_transform(lamb, beta, x, xi):
    N = x.size

    z_star = np.zeros(N)

    for i in range(N):
        val = x[i] - xi[i]/beta
        c = lamb/beta
        if np.abs(val) > c:
            z_star[i] = val - c
            if z_star[i] < 0:
                z_star[i] += 2 * c

    return z_star


if __name__ == "__main__":
    np.set_printoptions(precision=2)

    N = 100                # dimension
    STEPOUT = 5000          # max iterations
    RAND_SIZE = 2**10       # range of randomly drawn numbers
    GUESS_EFFORT = 100      # how much effort to guess beta 0
    TOLERANCE = 1e-2        # convergence criterium

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

    lamb = 2 * sigma * np.sqrt(np.log(N))
    y = (A @ w) + sigma * xi

    # oracles
    f = lambda x : 0.5 * autodot((A @ x) - y)
    df = lambda x : np.transpose(A) @ ((A @ x) - y)

    g = lambda x : lamb * np.linalg.norm(x, 1)


    # launch gradient descent
    w_recovered = argmin_F(f, df, g, lambda beta, x, xi: prox_transform(lamb, beta, x, xi), hyperpars)
    #print(w)
    #print(w_recovered)
    print("absolute difference from signal: " + str(np.linalg.norm(w - w_recovered, 1)))

    plt.show()
