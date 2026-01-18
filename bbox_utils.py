import numpy as np

from image_utils import cropbbox, meanvarpatchnorm

def bbox2rect(bbox):
    bbox = np.asarray(bbox)

    rect = np.zeros_like(bbox)
    if bbox.size:
        rect[:, 0] = np.min(bbox[:, [0, 2]], axis=1)
        rect[:, 1] = np.min(bbox[:, [1, 3]], axis=1)
        rect[:, 2] = np.max(bbox[:, [0, 2]], axis=1) - rect[:, 0]
        rect[:, 3] = np.max(bbox[:, [1, 3]], axis=1) - rect[:, 1]

    return rect

def bboxoverlapval(bb1, bb2, normtype=0):
    bb1 = np.asarray(bb1)
    bb2 = np.asarray(bb2)

    ov = np.zeros((bb1.shape[0], bb2.shape[0]))
    for i in range(bb1.shape[0]):
        for j in range(bb2.shape[0]):
            ov[i, j] = bboxsingleoverlapval(bb1[i], bb2[j], normtype)
    return ov


def bboxsingleoverlapval(bb1, bb2, normtype):
    bb1 = np.array([
        min(bb1[0], bb1[2]),
        min(bb1[1], bb1[3]),
        max(bb1[0], bb1[2]),
        max(bb1[1], bb1[3])
    ])

    bb2 = np.array([
        min(bb2[0], bb2[2]),
        min(bb2[1], bb2[3]),
        max(bb2[0], bb2[2]),
        max(bb2[1], bb2[3])
    ])

    if normtype < 0:
        ua = 1
    elif normtype == 1:
        ua = (bb1[2] - bb1[0] + 1) * (bb1[3] - bb1[1] + 1)
    elif normtype == 2:
        ua = (bb2[2] - bb2[0] + 1) * (bb2[3] - bb2[1] + 1)
    else:
        bu = np.array([
            min(bb1[0], bb2[0]),
            min(bb1[1], bb2[1]),
            max(bb1[2], bb2[2]),
            max(bb1[3], bb2[3])
        ])
        ua = (bu[2] - bu[0] + 1) * (bu[3] - bu[1] + 1)

    bi = np.array([
        max(bb1[0], bb2[0]),
        max(bb1[1], bb2[1]),
        min(bb1[2], bb2[2]),
        min(bb1[3], bb2[3])
    ])

    iw = bi[2] - bi[0] + 1
    ih = bi[3] - bi[1] + 1

    if iw > 0 and ih > 0:
        return (iw * ih) / ua
    return 0.0

def prunebboxes(bbox, conf, ovthresh):
    bbox = np.asarray(bbox)
    conf = np.asarray(conf)

    # sort by descending confidence
    isort = np.argsort(-conf)
    bbox = bbox[isort]
    conf = conf[isort]

    resbbox = []
    resconf = []

    freeflags = np.ones(len(conf), dtype=bool)

    while np.any(freeflags):
        indfree = np.where(freeflags)[0]
        indmax = indfree[np.argmax(conf[indfree])]

        ov = bboxoverlapval(bbox[indmax:indmax+1], bbox[indfree])[0]
        indsel = indfree[ov >= ovthresh]

        resbbox.append(bbox[indsel].mean(axis=0))
        resconf.append(conf[indmax])

        freeflags[indsel] = False

    return np.array(resbbox), np.array(resconf)

def winscandetsvm_onescale(img, model, wxsz, wysz, confthresh=-np.inf):
    img = np.asarray(img)
    ysz, xsz = img.shape[:2]

    # generate sliding windows
    x, y = np.meshgrid(np.arange(1, xsz - wxsz + 2),
                       np.arange(1, ysz - wysz + 2))
    bbox = np.stack([x.ravel(), y.ravel(),
                     x.ravel() + wxsz - 1,
                     y.ravel() + wysz - 1], axis=1)
    n = bbox.shape[0]

    print(f"evaluating {n} subwindows")

    # crop all windows
    imgcropall = np.zeros((wysz, wxsz, n))
    for i in range(n):
        imgcropall[:, :, i] = cropbbox(img, bbox[i, :])

    # normalize patches
    imgcropall = meanvarpatchnorm(imgcropall)

    X = imgcropall.reshape(wysz * wxsz, n)

    # run classification
    conf = (X.T @ model.W - model.b).ravel()

    ind = np.where(conf > confthresh)[0]
    bbox = bbox[ind, :]
    conf = conf[ind]
    normpatches = imgcropall[:, :, ind]

    return bbox, conf, normpatches
