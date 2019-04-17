#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Interpolation


# In[2]:


import numpy as np
import pylab as pl
import matplotlib.pyplot as plt
from scipy.interpolate.rbf import Rbf
from scipy.spatial import distance_matrix
from numpy.linalg import inv as inverse
from math import exp, pow


# In[3]:


def euclidean_norm(x1, x2):
    return np.sqrt(((x1 - x2)**2).sum(axis=0))


# In[4]:


def get_normal(x1, x2):
    norm = euclidean_norm
    if len(x1.shape) == 1:
        x1 = x1[np.newaxis, :]
    if len(x2.shape) == 1:
        x2 = x2[np.newaxis, :]
    x1 = x1[..., :, np.newaxis]
    x2 = x2[..., np.newaxis, :]
    return norm(x1, x2)


# In[5]:



def phi_sigma(s, sigma):
    #r = np.multiply(s, s)
    #return np.exp(-(r/(2 * sigma))**2)
    return np.divide(
                np.exp(-np.multiply(s, s)),
                (2.0 * pow(sigma, 2))
            )


# In[6]:



def get_ys(xs, x, w, sigma):
    dists = get_normal(xs,x)
    phs = phi_sigma(dists, sigma)
    return np.dot(phs, w)


# In[7]:



def interpolation_rbf(xs, x, y, sigma):
    # get inverse of similarity matrix for x and get weights
    x = x.reshape(x.size, 1)
    y = y.reshape(y.size, 1)
    inv_sim_matrix = inverse(phi_sigma(distance_matrix(x, x), sigma))
    weights = np.dot(inv_sim_matrix, y)

    return weights


# In[8]:


def scipy_fun_plot(n, x, y, xs, epsilon_arr, img_name):
    

    # draw plot here
    fig = pl.figure(figsize=(20, 5))

    draw = fig.add_subplot(111)
    draw.set_title("Radial Basis Function Interpolation")
    draw.set_ylim([-2, 4])
    draw.set_xlim([-0.5, 20.5])
    draw.plot(x, y, 'bo', label="data")
    draw.plot(x, y, 'r', label="linear")
    
    
    plot_lines = [ 'orange', 'gray', 'brown', 'black' ]
    function_name='gaussian'
    
    
    for indx in range(0, 4, 1):
        rbf = Rbf(x, y, function=function_name, epsilon=epsilon_arr[indx])
        bim = rbf(xs)
        
        draw.plot(xs, bim, plot_lines[indx], linewidth=1.3, label=r"$sigma$="+str(indx))

    draw.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0.0)
    draw.plot()

    pl.savefig(img_name, bbox_inches='tight')
    pl.show()


# In[9]:


def local_implementation(sigma, img_name):
# -----------------------------------------------------------------------------
# local implementation
    fig2 = pl.figure(figsize=(20, 5))

    draw = fig2.add_subplot(111)
    draw.set_title("Radial Basis Function Interpolation")
    draw.set_ylim([-2, 4])
    draw.set_xlim([-0.5, 20.5])
    draw.plot(x, y, 'bo', label="data", markersize=8)
    draw.plot(x, y, 'r', label="linear")

    
    weights = interpolation_rbf(xs, x, y, sigma)
    bimes = get_ys(xs, x, weights, sigma)
    draw.plot(xs, bimes, 'orange', linewidth=3, label=r"$sigma$={}, own".format(sigma))

    rbf = Rbf(x, y, function='gaussian', epsilon=0.5)
    bim = rbf(xs)
    draw.plot(xs, bim, 'black', linewidth=1.1, label=r"$sigma$={}, Scipy".format(sigma))

    draw.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=6, mode="expand", borderaxespad=0.0)
    draw.plot()

    pl.savefig(img_name, bbox_inches='tight')
    pl.show()


# In[10]:


n = 20
x = np.arange(n) + np.random.randn(n) * 0.2
y = np.random.rand(n) * 2
# we will apply the function to this data
xs = np.linspace(0, n, 100)

epsilon_arr = [ 0.5, 1, 2, 4 ]
img_name = 'scipy_rbf_interpolation.png'
scipy_fun_plot(n,x,y,xs,epsilon_arr, img_name)


# In[11]:


n = 20
x = np.arange(n) + np.random.randn(n) * 0.2
y = np.random.rand(n) * 2
# we will apply the function to this data
xs = np.linspace(0, n, 100)

epsilon_arr = [ 2, 4, 6, 8 ]
img_name = 'scipy_rbf_interpolation_1.png'
scipy_fun_plot(n,x,y,xs,epsilon_arr, img_name)


# In[12]:


n = 20
x = np.arange(n) + np.random.randn(n) * 0.2
y = np.random.rand(n) * 2
# we will apply the function to this data
xs = np.linspace(0, n, 100)

epsilon_arr = [ 1, 2, 3, 4 ]
img_name = 'scipy_rbf_interpolation_2.png'
scipy_fun_plot(n,x,y,xs,epsilon_arr, img_name)


# In[13]:


sigma = 0.5
img_name = 'local_rbf_interpolation.png'
local_implementation(sigma, img_name)


# In[14]:


sigma = 1
img_name = 'local_rbf_interpolation_1.png'
local_implementation(sigma, img_name)


# In[15]:


sigma = 2
img_name = 'local_rbf_interpolation_2.png'
local_implementation(sigma, img_name)


# In[ ]:





# In[ ]:




