import scipy as sp
import scipy.signal as sg
import numpy as np
import pylab as pyl
import scipy.misc as msc
import scipy.ndimage as ndimage
import matplotlib as mpl
import matplotlib.pyplot as plt


def createGaussianFilter( sigma=1.0, sigmax=None, sigmay=None, size=None ):

    sigmax = sigmax or sigma
    sigmay = sigmay or sigma

    size = size or (int( np.ceil( sigmax * 2.575 ) * 2 + 1 ), int( np.ceil( sigmay * 2.575 ) * 2 + 1 ))
    
    msize = size[0]
    x = np.arange( msize )
    gx = np.exp( -0.5 * ( ( x-msize/2) / sigmax ) ** 2 )
    gx /= gx.sum()

    nsize = size[1]
    y = np.arange( nsize )
    gy = np.exp( -0.5 * ( ( y-nsize / 2 ) / sigmay ) ** 2 )
    gy /= gy.sum()

    G = np.outer( gx, gy )
    G /= G.sum()
    return G, gx, gy



def createGaussianFilterBySize( size=(3,3) ):
    sigmas = [((s-1)/2)/2.575 for s in size]
    return createGaussianFilter(sigmax = sigmas[0], sigmay=sigmas[1], size=size)

def polartocartesian(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)

def cartesiantopolar(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def transform_xy(Im):
    (sy,sx) = Im.shape
    mu = [sy/2,sx/2]
    #mu = [180,180]
    Final = np.zeros((sy,sx))
    thr = 2*np.pi
    r_i = np.linspace(0,np.sqrt(mu[1]**2 + mu[0]**2), sy)
    theta_i = np.linspace(0, thr, sy)
    r_grid, theta_grid = np.meshgrid(r_i, theta_i)
    _, gy, gx = createGaussianFilterBySize(size=(21,21))
    xo, yo = polartocartesian(r_grid,theta_grid)
    xo = xo + mu[1]
    yo = yo + mu[0]
    C = np.vstack((yo.flatten(),xo.flatten()))

    transformed = ndimage.interpolation.map_coordinates(Im,C).reshape(sy,sx)
    transformed_blur = ndimage.convolve1d(transformed,gx,axis=0).T[::-1,:]

    mul_fac = sx/np.sqrt(mu[1]**2 + mu[0]**2)
    ys = mu[0] - pyl.arange(sy)
    xs = mu[1] - pyl.arange(sx)
    xq, yq = np.meshgrid(xs,ys)
    r,t = cartesiantopolar(yq,xq)
    inx = abs(sx - t * (sx/thr))%(sx-1)
    iny = abs(sy - r*mul_fac)
    X = np.vstack((iny.flatten(),inx.flatten()))
    final = ndimage.interpolation.map_coordinates(transformed_blur,X).reshape(sy,sx).T[:,::-1]

    return transformed, transformed_blur, final

face = msc.imread("bauckhage.jpg", flatten=True).astype('float')
clock = msc.imread("clock.jpg", flatten=True).astype('float')
flag = msc.imread("flag.png", flatten=True).astype('float')

result = transform_xy(flag)
plt.figure()
plt.imshow(result[0], cmap="Greys_r")
msc.imsave("resulting_images/one_.png", result[0])
plt.show()

plt.imshow(result[1], cmap="Greys_r")
msc.imsave("resulting_images/two_.png", result[1])
plt.show()

plt.imshow(result[2], cmap="Greys_r")
msc.imsave("resulting_images/fin_.png", result[2])
plt.show()
