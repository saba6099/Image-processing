#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy.misc as msc
#from functions import *
import time as time
import scipy.ndimage as ndimg
from scipy.spatial.distance import euclidean
from cmath import sqrt
from math import cos, sin, pow, e, pi
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import scipy.misc as msc
import time as time
import scipy.ndimage as ndimg


# In[2]:


def plot_time_line(title,arrVal,new_img, cmap, x,y,z, figure):
    print("Time: ", title, arrVal)    
    figure.add_subplot(x,y,z).set_title(title)
    plt.imshow(new_img, cmap)
              


# In[3]:



#gaussian filter
def gaussianFilter( sigma=1.0, sigmax=None, sigmay=None, size=None ):

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

#naive_convolution1D function
def naiveConvolve1d( img, filter1d ):
    
    width = img.shape[0]
    height = img.shape[1]
    mask = len( filter1d )
    bound = np.floor( mask / 2 )
    new_image = np.ndarray( ( width, height ) )
    for y in range( 0, height ):
        for x in range( 0, width ):
            sums = 0.0
            for i in range( 0, mask ):
                xdash = int( x + ( i - bound ) )
                if( 0 > xdash or width <= xdash ):
                    sums += 0.0
                else:
                    sums += img[ xdash, y ] * filter1d[ i ]
            new_image[ x, y ] = sums
    return new_image

#naive convolve2D
def naiveConvolve2d( img, filter2d ):

    width = img.shape[0]
    height = img.shape[1]
    m = filter2d.shape[0]
    n = filter2d.shape[1]
    boundm = np.floor( m / 2 )
    boundn = np.floor( n / 2 )
    new_image = np.ndarray( ( width, height ) )
    for x in range( 0, width ):
        for y in range( 0, height ):
            sums = 0.0
            for i in range( 0, m ):
                for j in range( 0, n ):
                    xdash = int( x + ( i - boundm ) )
                    ydash = int( y + ( j - boundn ) )
                    if( 0 > xdash or
                        width <= xdash or
                        0 > ydash or
                        height <= ydash ):
                        sums += 0.0
                    else:
                        sums += img[ xdash, ydash ] * filter2d[ i, j ]
            new_image[ x, y ] = sums
    return new_image

def convolve2dFFT( img, size ):
    sigmas = [((s-1)/2)/2.575 for s in size]
    G,_,_ = gaussianFilter(sigmax = sigmas[0], sigmay=sigmas[1], size=img.shape)
    Fg = fft.fft2(G)
    Fi = fft.fft2(img)
    pm = np.multiply(Fg,Fi)
    ipm = np.abs(fft.ifftshift(fft.ifft2(pm)))
    return ipm


# In[4]:



def task1(img):
    """
    """
    
    sizes = range(3,25,2)
    
    const_val = 2.575
    sigmas = [((size-1)/2)/2.575 for size in sizes]
    nc2 = nc1 = cfft = sysc= np.ndarray(len(sizes))
    
    cmap="Greys_r"

    for i in range(len(sizes)):
        # G, gx, gy = createGaussianFilter(sigma=sigmas[len(sigmas) - 1])
        figure = plt.figure("{}x{}, sig={:.3f}".format(sizes[i],sizes[i],sigmas[i]))
        print("{}x{}, sig={:.3f}".format(sizes[i],sizes[i],sigmas[i]))
        figure.add_subplot(2,3,1).set_title("original")
        plt.imshow(img, cmap)
        
        print("-----------------------------------------")
                
        title = "NaiveConv1D"
        G, gx, gy = gaussianFilter(sigma=sigmas[i])
        start_time = time.time()
        new_img = naiveConvolve1d(img.T, gx)
        new_img = naiveConvolve1d(new_img.T, gy)       
        end_time = time.time()
        nc1[i] = end_time-start_time
        
        plot_time_line(title, nc1[i], new_img, cmap, 2,3,2, figure)
        
                
        title = "NaiveConv2D"
        start_time = time.time()
        new_img = naiveConvolve2d(img, G)
        end_time = time.time()
        nc2[i] = end_time-start_time
        plot_time_line(title, nc2[i], new_img, cmap, 2,3,3, figure)
        
                    
        title = "ConFFT"
        start_time = time.time()
        new_img = convolve2dFFT(img, (sizes[i],sizes[i]))
        end_time = time.time()
        cfft[i] = end_time-start_time
        plot_time_line(title, cfft[i], new_img, cmap, 2,3,4, figure)
        
    
        title = "SciPy Con"
        start_time = time.time()
        new_img = ndimg.convolve(img, G, mode="constant", cval=0.0)
        end_time = time.time()
        sysc[i] = end_time-start_time
        plot_time_line(title, sysc[i], new_img, cmap, 2,3,5, figure)
        
        
        print("-----------------------------------------")
        plt.savefig("resulting_images/{}x{}, sig={:.3f}.png".format(sizes[i],sizes[i],sigmas[i]), bbox_inches='tight')

        fig = plt.figure()
        ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
        ax.plot(sizes, np.log(nc1*1000), 'k--', lw=2, label='nc1')
        ax.plot(sizes, np.log(nc2*1000), 'k:', lw=2, label='nc2')
        ax.plot(sizes, np.log(cfft*1000), 'r--', lw=2, label='cfft')
        ax.plot(sizes, np.log(sysc*1000), 'r:', lw=2, label='sysc')
        ax.set_xlabel('Filter Size', size=22)
        ax.set_ylabel('Log(time*1000)', size=22)
        ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.savefig("resulting_images/task1plots.png".format(sizes[i],sizes[i],sigmas[i]), bbox_inches='tight')


# In[5]:


def readImg(name):
    return msc.imread(name).astype("float")


# In[ ]:


face = readImg("images/bauckhage.jpg")
clock = readImg("images/clock.jpg")

task1(face)


# In[ ]:





# In[ ]:




