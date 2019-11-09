import matplotlib.pyplot as plt
import matplotlib.patches as pltpatch  # for Arc
import matplotlib.collections as pltcoll

import numpy as np


def draw(Vs, W, inputNames=None, outputNames=None, gray=False):

    def isOdd(x):
        return x % 2 != 0

    W = Vs + [W]
    nLayers = len(W)

    # calculate xlim and ylim for whole network plot
    #  Assume 4 characters fit between each wire
    #  -0.5 is to leave 0.5 spacing before first wire
    xlim = max(map(len, inputNames))/4.0 if inputNames else 1
    ylim = 0

    for li in range(nLayers):
        ni, no = W[li].shape  # no means number outputs this layer
        if not isOdd(li):
            ylim += ni + 0.5
        else:
            xlim += ni + 0.5

    ni, no = W[nLayers-1].shape
    if isOdd(nLayers):
        xlim += no + 0.5
    else:
        ylim += no + 0.5

    # Add space for output names
    if outputNames:
        if isOdd(nLayers):
            ylim += 0.25
        else:
            xlim += round(max(map(len,outputNames))/4.0)

    ax = plt.gca()

    character_width_factor = 0.07
    padding = 2
    if inputNames:
        x0 = max([1, max(map(len, inputNames)) * (character_width_factor * 3.5)])
    else:
        x0 = 1
    y0 = 0  # to allow for constant input to first layer
    # First Layer
    if inputNames:
        y = 0.55
        for n in inputNames:
            y += 1
            ax.text(x0 - (character_width_factor * padding), y, n, horizontalalignment="right", fontsize=20)

    patches = []
    for li in range(nLayers):
        thisW = W[li]
        maxW = np.max(np.abs(thisW))
        ni, no = thisW.shape
        if not isOdd(li):
            # Even layer index. Vertical layer. Origin is upper left.
            # Constant input
            ax.text(x0-0.2, y0+0.5, '1', fontsize=20)
            for i in range(ni):
                ax.plot((x0, x0+no-0.5), (y0+i+0.5, y0+i+0.5), color='gray')
            # output lines
            for i in range(no):
                ax.plot((x0+1+i-0.5, x0+1+i-0.5), (y0, y0+ni+1), color='gray')
            # cell "bodies"
            xs = x0 + np.arange(no) + 0.5
            ys = np.array([y0+ni+0.5]*no)
            for x, y in zip(xs, ys):
                patches.append(pltpatch.RegularPolygon((x, y-0.4), 3, 0.3, 0, color ='#555555'))
            # weights
            if gray:
                colors = np.array(["black", "gray"])[(thisW.flat >= 0)+0]
            else:
                colors = np.array(["red", "green"])[(thisW.flat >= 0)+0]
            xs = np.arange(no) + x0+0.5
            ys = np.arange(ni) + y0 + 0.5
            coords = np.meshgrid(xs, ys)
            for x, y, w, c in zip(coords[0].flat, coords[1].flat,
                                  np.abs(thisW/maxW).flat, colors):
                patches.append(pltpatch.Rectangle((x-w/2, y-w/2), w, w, color=c))
            y0 += ni + 1
            x0 += -1  # shift for next layer's constant input
        else:
            # Odd layer index. Horizontal layer. Origin is upper left.
            # Constant input
            ax.text(x0+0.5, y0-0.2, '1', fontsize=20)
            # input lines
            for i in range(ni):
                ax.plot((x0+i+0.5,  x0+i+0.5), (y0, y0+no-0.5), color='gray')
            # output lines
            for i in range(no):
                ax.plot((x0, x0+ni+1), (y0+i+0.5, y0+i+0.5), color='gray')
            # cell "bodies"
            xs = np.array([x0 + ni + 0.5] * no)
            ys = y0 + 0.5 + np.arange(no)
            for x, y in zip(xs, ys):
                patches.append(pltpatch.RegularPolygon((x-0.4, y), 3, 0.3, -np.pi/2, color ='#555555'))
            # weights
            if gray:
                colors = np.array(["black", "gray"])[(thisW.flat >= 0)+0]
            else:
                colors = np.array(["red", "green"])[(thisW.flat >= 0)+0]
            xs = np.arange(ni)+x0 + 0.5
            ys = np.arange(no)+y0 + 0.5
            coords = np.meshgrid(xs, ys)
            for x, y, w, c in zip(coords[0].flat, coords[1].flat,
                                  np.abs(thisW/maxW).flat, colors):
                patches.append(pltpatch.Rectangle((x-w/2, y-w/2), w, w, color=c))
            x0 += ni + 1
            y0 -= 1  # shift to allow for next layer's constant input

    collection = pltcoll.PatchCollection(patches, match_original=True)
    ax.add_collection(collection)

    # Last layer output labels
    if outputNames:
        if isOdd(nLayers):
            x = x0+1.5
            for n in outputNames:
                x += 1
                ax.text(x, y0+0.5, n, fontsize=20)
        else:
            y = y0+0.6
            for n in outputNames:
                y += 1
                ax.text(x0+0.2, y, n, fontsize=20)
    ax.axis([0, xlim, ylim, 0])
    ax.axis('off')
