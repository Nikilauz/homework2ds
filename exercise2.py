# disclaimer: this file was transpiled from MATLAB using ChatGPT
#             to validate it, there are tests in the test folder

##############################################################################
#                                                                            #
# Practical assignment: Face detection                                       #
#                                                                            #
##############################################################################
#                                                                            #
#   Part 2: scanning window face detection with non-max suppression          #
#                                                                            #
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from bbox_utils import prunebboxes
from image_utils import cropbbox, meanvarpatchnorm
from imageio import imread
from visualize import showbbox, showimage

imgpath = "images"
respath = "results"

plt.gray()

##############################################################################
############################ load test image #################################
##############################################################################

imgfname = "img1.jpg"
img = imread(f"{imgpath}/{imgfname}")

showimage(img)
plt.title(imgfname)
plt.show()

# convert to grayscale by averaging channels (MATLAB: mean(img,3))
gimg = img.mean(axis=2)

##############################################################################
################### decompose the image into sub-windows ######################
##############################################################################

ysz, xsz, csz = img.shape
wxsz = 24
wysz = 24

# MATLAB: [x,y]=meshgrid(1:xsz-wxsz,1:ysz-wysz);
x, y = np.meshgrid(
    np.arange(1, xsz - wxsz + 1),
    np.arange(1, ysz - wysz + 1)
)

bbox = np.stack([
    x.ravel(),
    y.ravel(),
    x.ravel() + wxsz - 1,
    y.ravel() + wysz - 1
], axis=1)

n = bbox.shape[0]

imgcropall = np.zeros((wysz, wxsz, n))
for i in range(n):
    imgcropall[:, :, i] = cropbbox(gimg, bbox[i, :])

##############################################################################
################### normalize and reshape test patches #######################
######### Linear classifier can be evaluated efficiently by
######### using dot-product between image patches and W. Before
######### computing the confidence we need to normalize and reshape
######### test image patches.
##############################################################################

imgcropall = meanvarpatchnorm(imgcropall)

# MATLAB: X=transpose([reshape(imgcropall,wysz*wxsz,n)]);
X = imgcropall.reshape(wysz * wxsz, n).T

# Linear classifier confidence

# TODO feed results from training!
Wbest = np.zeros(24*24)
bbest = 0
conf = X @ Wbest + bbest

##############################################################################
################### display most confident detections #########################
##############################################################################

ndisp = 30
idx = np.argsort(conf)[::-1]

plt.figure()
showimage(img)
showbbox(bbox[idx[:ndisp], :])
plt.title(f"{ndisp} best detections")
plt.show()

input("press a key...")

##############################################################################
##############################################################################
################# Non-maxima suppression of multiple responses ################
##############################################################################
##############################################################################

# Scanning-window style classification of image patches typically
# results in many multiple responses around the target object.
# A standard practice to deal with this is non-maxima suppression (NMS).

confthresh = 4.5
indsel = np.where(conf > confthresh)[0]

nmsbbox, nmsconf = prunebboxes(
    bbox[indsel, :],
    conf[indsel],
    ovthresh=0.2
)

##############################################################################
################ display detections after NMS #################################
##############################################################################

confthreshnms = 1
plt.figure()
showimage(img)

indsel = np.where(nmsconf > confthreshnms)[0]
labels = [f"{c:.2f}" for c in nmsconf[indsel]]

showbbox(
    nmsbbox[indsel, :],
    color=(1, 1, 0),
    labels=labels
)

plt.title(
    f"{len(indsel)} NMS detections above threshold {confthreshnms:.3f}"
)
plt.show()
