# disclaimer: this file was transpiled from MATLAB using ChatGPT
#             to validate it, there are tests in the test folder

from matplotlib import pyplot as plt
import matplotlib.patches as patches
import numpy as np

from bbox_utils import bbox2rect

def showbbox(bbox, ccol=None, labels=None, linewidth=3, textheight=None):
    bbox = np.asarray(bbox)
    n = bbox.shape[0]

    if labels is None:
        labels = [None] * n
    if textheight is None:
        textheight = 5 + linewidth

    if ccol is None or len(ccol) == 0:
        # MATLAB lines(n) equivalent
        import matplotlib.pyplot as plt
        ccol = plt.cm.tab10(np.linspace(0, 1, n))[:, :3]

    ccol = np.asarray(ccol)
    if ccol.ndim == 1:
        ccol = np.tile(ccol, (n, 1))

    showrect(bbox2rect(bbox), ccol, labels, linewidth, textheight)

def showimage(img, ah=None, grayflag=None, xlim=None, ylim=None):
    img = np.squeeze(np.asarray(img)).astype(float)
    h, w = img.shape[:2]

    if xlim is None:
        xlim = [1, w]
    if ylim is None:
        ylim = [1, h]
    if ah is None:
        ah = plt.gca()

    xloc = np.linspace(xlim[0], xlim[1], w)
    yloc = np.linspace(ylim[0], ylim[1], h)

    # grayscale
    if grayflag is not None or img.ndim < 3:
        gimg = img.mean(axis=2) if img.ndim == 3 else img

        if grayflag is not None and len(grayflag) >= 3:
            gmin, gmax = grayflag[1], grayflag[2]
        else:
            gmin, gmax = gimg.min(), gimg.max()

        gres = 256
        range_val = max(np.finfo(float).eps, gmax - gmin)
        gimg_scaled = 1 + (gres / range_val) * (gimg - gmin)

        colormap = np.linspace(0, 1, gres)
        colormap_rgb = np.stack([colormap, colormap, colormap], axis=1)

        imh = ah.imshow(gimg_scaled, extent=[xloc[0], xloc[-1], yloc[0], yloc[-1]],
                        cmap=plt.cm.gray, origin='upper')

    else:  # color image
        minval, maxval = img.min(), img.max()
        range_val = max(np.finfo(float).eps, maxval - minval)
        img_scaled = (img - minval) / range_val
        imh = ah.imshow(img_scaled, extent=[xloc[0], xloc[-1], yloc[0], yloc[-1]],
                        origin='upper')

    ah.set_aspect('equal')
    ah.axis('off')
    return imh

def showrect(rect, ccol=None, labels=None, linewidth=1, textheight=None, cornersflag=0):
    rect = np.asarray(rect)
    n = rect.shape[0]

    if textheight is None:
        textheight = 5 + linewidth
    if ccol is None or len(ccol) == 0:
        ccol = plt.cm.tab10(np.linspace(0, 1, n))[:, :3]
    ccol = np.asarray(ccol)
    if ccol.ndim == 1:
        ccol = np.tile(ccol, (n, 1))
    if labels is None:
        labels = [None] * n

    ax = plt.gca()
    for i in range(n):
        if rect[i, 2] != 0 and rect[i, 3] != 0:
            x1, y1, w, h = rect[i]

            if not cornersflag:
                ax.add_patch(patches.Rectangle((x1, y1), w, h,
                                           edgecolor=ccol[i],
                                           fill=False,
                                           linewidth=linewidth))
            else:
                len_corner = np.mean([w, h]) / 5
                x2, y2 = x1 + w, y1 + h

                # Draw 8 corner lines
                ax.plot([x1, x1+len_corner], [y1, y1], color=ccol[i], linewidth=linewidth)
                ax.plot([x2, x2-len_corner], [y1, y1], color=ccol[i], linewidth=linewidth)
                ax.plot([x1, x1], [y1, y1+len_corner], color=ccol[i], linewidth=linewidth)
                ax.plot([x2, x2], [y1, y1+len_corner], color=ccol[i], linewidth=linewidth)
                ax.plot([x1, x1+len_corner], [y2, y2], color=ccol[i], linewidth=linewidth)
                ax.plot([x2, x2-len_corner], [y2, y2], color=ccol[i], linewidth=linewidth)
                ax.plot([x1, x1], [y2, y2-len_corner], color=ccol[i], linewidth=linewidth)
                ax.plot([x2, x2], [y2, y2-len_corner], color=ccol[i], linewidth=linewidth)

            if labels[i] is not None:
                ax.text(x1, y1 - 5, str(labels[i]),
                        color=ccol[i],
                        fontsize=10,
                        fontweight='bold')