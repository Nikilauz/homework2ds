import numpy as np

def cropbbox(img, bbox):
    bbox = np.round(bbox).astype(int)
    ysz, xsz = img.shape[:2]
    csz = 1 if img.ndim == 2 else img.shape[2]

    # inside image → simple crop
    if (bbox[0] >= 1 and bbox[2] <= xsz and
        bbox[1] >= 1 and bbox[3] <= ysz):

        return img[bbox[1]-1:bbox[3], bbox[0]-1:bbox[2], ...]

    # outside image → mirrored padding
    x, y = np.meshgrid(
        np.arange(bbox[0], bbox[2] + 1),
        np.arange(bbox[1], bbox[3] + 1)
    )

    go = True
    while go:
        go = False

        ix = x < 1
        iy = y < 1
        x[ix] = np.abs(x[ix]) + 2
        y[iy] = np.abs(y[iy]) + 2

        ix = x > xsz
        iy = y > ysz
        if np.any(ix):
            x[ix] = 2 * xsz - x[ix] - 1
            go = True
        if np.any(iy):
            y[iy] = 2 * ysz - y[iy] - 1
            go = True

    # convert to 0-based indexing
    x -= 1
    y -= 1

    # gather pixels
    flat = img.reshape(ysz * xsz, -1)
    ind = y.ravel() * xsz + x.ravel()
    out = flat[ind]

    out_shape = x.shape + (() if img.ndim == 2 else (img.shape[2],))
    return out.reshape(out_shape)

def meanvarpatchnorm(images):
    ysz, xsz, nsz = images.shape

    imgs = images.reshape(ysz * xsz, nsz)
    meanval = imgs.mean(axis=0)
    stdval = imgs.std(axis=0, ddof=0)

    imgs = imgs - meanval
    imgs = imgs / (np.finfo(float).eps + stdval)

    return imgs.reshape(ysz, xsz, nsz)