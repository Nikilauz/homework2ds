# disclaimer: this file was transpiled from MATLAB using ChatGPT
#             to validate it, there are tests in the test folder

"""
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                            %
% Practical assignment: Face detection                                       %
%                                                                            %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%                                                                            %
%   Part 1:  load and prepare image the data                                 %
%                                                                         %
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
"""

import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

from gradient_descent import argmin_F, autodot
from image_utils import meanvarpatchnorm
from lasso_recovery import prox_transform
from visualize import showimage

# ---------------------------
# Paths
# ---------------------------
imgpath = "faces/images"
respath = "results"

# colormap(gray) equivalent in matplotlib
plt.gray()

# ---------------------------
# Parameters: configurable training / validation sizes
# ---------------------------
ntrain_pos = 1000
ntrain_neg = 1000

# ---------------------------
# Part 1: Loading and normalizing training images
# ---------------------------

# Load training images
posfname = f"{imgpath}/possamples.mat"
negfname = f"{imgpath}/negsamples.mat"

posdata = loadmat(posfname)["possamples"].astype(float)
negdata = loadmat(negfname)["negsamples"].astype(float)

npos, nneg = posdata.shape[2], negdata.shape[2]
print(f"Loaded {npos} positive samples from {posfname}")
print(f"Loaded {nneg} negative samples from {negfname}\n")

"""
%%%%%%%%%%%%%%%% positive and negative training images are now 
%%%%%%%%%%%%%%%% inside 'possamples' and 'negsamples' 3D arrays
%%%%%%%%%%%%%%%% of the size 24x24xN where
%%%%%%%%%%%%%%%%  - 24x24 is the pixel size of cropped faces
%%%%%%%%%%%%%%%%  - N is the number of samples
%%%%%%%%%%%%%%%%"""


# ---------------------------
# Normalize pixel values to zero mean and unit variance
# ---------------------------
posdata = meanvarpatchnorm(posdata)
negdata = meanvarpatchnorm(negdata)

# ---------------------------
# Flatten training images into vectors (one sample per row)
# ---------------------------
ysz, xsz = posdata.shape[0], posdata.shape[1]
Xpos = posdata.reshape(ysz * xsz, npos).T
Xneg = negdata.reshape(ysz * xsz, nneg).T

# ---------------------------
# Make vector with sample labels
# +1 for positives, -1 for negatives
# ---------------------------
ypos = np.ones(npos)
yneg = -np.ones(nneg)

# ---------------------------
# Separate data into training and validation sets
# ---------------------------
Xtrain = np.vstack([Xpos[:ntrain_pos], Xneg[:ntrain_neg]])
ytrain = np.hstack([ypos[:ntrain_pos], yneg[:ntrain_neg]])

Xval = np.vstack([Xpos[ntrain_pos:2*ntrain_pos], Xneg[ntrain_neg:2*ntrain_neg]])
yval = np.hstack([ypos[ntrain_pos:2*ntrain_pos], yneg[ntrain_neg:2*ntrain_neg]])

print(f"Training set: {Xtrain.shape[0]} samples")
print(f"Validation set: {Xval.shape[0]} samples")

############## end of transpilation

Xtrain_flat = [x.ravel() for x in Xtrain]
Xval_flat = [x.ravel() for x in Xval]

# define M1 and M2

N = 24*24

L = lambda gamma, x_and_t, u, y : gamma * np.sum([np.log(1 + np.exp((1/gamma) * (1 - y[i]*(x_and_t[:-1] @ u[i] + x_and_t[-1])))) for i in range(N)])

dL = lambda gamma, x_and_t, u, y : gamma * np.sum([np.exp((1./gamma) * (1 - y[i](x_and_t[:-1] @ u[i] + x_and_t[-1]))) * (1./gamma) * (list(u[i]) + [1.])/(1 + np.exp((1./gamma) * (1 - y[i](x_and_t[:-1] @ u[i] + x_and_t[-1])))) for i in range(N)])

g_m1 = lambda kappa, x_and_t : kappa * np.linalg.norm(x_and_t[:-1], 1)

g_m2 = lambda lamb, x_and_t : 0.5 * lamb * autodot(x_and_t[:-1])

prox_m2 = lambda _ : np.zeros(N + 1)

# grid search

hyperpars = {
    "N": N+1,
    "STEPOUT": 1000,
    "RAND_SIZE": 10,
    "GUESS_EFFORT": 50,
    "TOLERANCE": 0.01
}

gammas = [1.]
kappas = [1.]

for gam in gammas:
    for kap in kappas:
        x_and_t_star_m1 = argmin_F(
            lambda x_and_t : L(gam, x_and_t, Xtrain_flat, ytrain),
            lambda x_and_t : dL(gam, x_and_t, Xtrain_flat, ytrain),
            lambda x_and_t : g_m1(kap, x_and_t),
            prox_transform,
            hyperpars
        )

        x_and_t_star_m2 = argmin_F(
            lambda x_and_t : L(gam, x_and_t, Xtrain_flat, ytrain),
            lambda x_and_t : dL(gam, x_and_t, Xtrain_flat, ytrain),
            lambda x_and_t : g_m2(kap, x_and_t),
            prox_m2,
            hyperpars
        )
