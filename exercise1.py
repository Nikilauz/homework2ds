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

from image_utils import meanvarpatchnorm
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
# Display a few positive and negative samples
# ---------------------------
possampleimage = []
negsampleimage = []
for _ in range(10):
    indpos = np.random.permutation(npos)[:10]
    indneg = np.random.permutation(nneg)[:10]

    # Reshape like MATLAB shiftdim + reshape
    posfaces = posdata[:, :, indpos].transpose(2, 0, 1).reshape(24*10, 24)
    negfaces = negdata[:, :, indneg].transpose(2, 0, 1).reshape(24*10, 24)

    possampleimage.append(posfaces)
    negsampleimage.append(negfaces)

possampleimage = np.hstack(possampleimage)
negsampleimage = np.hstack(negsampleimage)

print("Showing images of random Positive samples")
showimage(possampleimage.T)
plt.title("positive samples")
plt.show()

print("Showing images of random Negative samples")
showimage(negsampleimage.T)
plt.title("negative samples")
plt.show()

# ---------------------------
# Display average face and non-face images
# ---------------------------
avg_pos = np.mean(posdata[:, :, :min(2000, npos)], axis=2)
avg_neg = np.mean(negdata[:, :, :min(2000, nneg)], axis=2)

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
showimage(avg_pos)
plt.title("average positive image")
plt.subplot(1, 2, 2)
showimage(avg_neg)
plt.title("average negative image")
plt.show()

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
