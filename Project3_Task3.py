# -*- coding: utf-8 -*-
"""
Created on Sun Jan 20 18:15:25 2019

@author: Hirra
"""

import scipy as sp
import numpy as np
import pylab as pyl
import scipy.misc as msc
import scipy.ndimage as ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt


def anamorphosis(Im, r_in):

    dpi = 72
    r_cylinder = int(r_in*dpi);
    (sy,sx) = Im.shape
    Im = np.pad(Im,(0,r_cylinder), mode="constant")[:,:sx]
    (sy,sx) = Im.shape

    th_range = (pyl.pi)*2
    max_r = sy
    fy = 2*max_r
    fx = 2*max_r
    #Mid
    ys = fy/2 - pyl.arange(fy)
    xs = fx/2 - pyl.arange(fx)
    
    xq, yq = np.meshgrid(ys,xs)
    rs = pyl.sqrt(yq*yq+xq*xq)
    rsmx = rs < max_r
    ths = pyl.arctan2(yq,xq)
    thsmx = ths < th_range
    inx = abs(sx - ths * (sx/th_range))%(sx-1)
    inx[~thsmx] = sx*2
    iny = abs(sy - rs)
    iny[~rsmx] = sy*2
    X = np.vstack((iny.flatten(),inx.flatten()))
    Final = ndimage.interpolation.map_coordinates(Im,X).reshape(fy,fx)[::-1,:]

    return Final


prof = msc.imread("bauckhage.jpg", flatten=True).astype('float')
clock = msc.imread("clock.jpg", flatten=True).astype('float')


plt.figure()

msc.imsave("original.png", prof)

#Task1


result = anamorphosis(prof, 0.0)
plt.imshow(result, cmap="Greys_r")
msc.imsave("zero.png", result)
plt.show()

#Task2


result = anamorphosis(clock, 2.0)
plt.imshow(result, cmap="Greys_r")
msc.imsave("donuts_1.png", result)
plt.show()

result = anamorphosis(clock, 4.0)
plt.imshow(result, cmap="Greys_r")
msc.imsave("donuts_2.png", result)
plt.show()

result = anamorphosis(clock, 6.0)
plt.imshow(result, cmap="Greys_r")
msc.imsave("donuts_3.png", result)
plt.show()

result = anamorphosis(clock, 8.0)
plt.imshow(result, cmap="Greys_r")
msc.imsave("donuts_4.png", result)
plt.show()