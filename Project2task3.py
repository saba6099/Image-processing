import numpy as np
import matplotlib.pyplot as plt
import scipy.misc as msc
import time as time
from cmath import sqrt
from skimage import io
from math import cos, sin, e, pi

def readIntensityImage(filename):
    f = io.imread(filename).astype("float")
    return f

def causal_coeff(sigma):
    # precomputed values of alpha, beta, gamma, omega (for i = 1, 2)
    a1, a2 = 1.68, -0.6803
    b1, b2 = 3.7350, -0.2598
    g1, g2 = 1.7830, 1.7230
    w1, w2 = 0.6318, 1.9970

    aplus = np.zeros(4)
    bplus = np.zeros(4)

    # a0

    aplus[0] = a1 + a2

    # a1
    aplus[1] = (e * (-(g2 / sigma))) * (b2 * sin(w2/sigma) - (a2 + 2*a1) * cos(w2/sigma)) + (e * (-(g1 / sigma))) * (b1 * sin(w1/sigma) - (2*a2 + a1) * cos(w1/sigma))

    # a2
    aplus[2] = 2 * (e ** (-((g1 + g2) / sigma))) * ((a1 + a2) * cos(w2/sigma) * cos(w1/sigma) - cos(w2/sigma) * b1 * sin(w1/sigma) \
                                           - cos(w1/sigma) * b2 * sin(w2/sigma)) + a2 * (e * (-2 * (g1 / sigma))) + a1 * (e * (-2 * (g2 / sigma)))

    # a3
    aplus[3] = (e * (-((g2 + 2 * g1) / sigma))) * (b2 * sin(w2/sigma) - a2 * cos(w2/sigma)) + (e * (-((g1 + 2 * g2) / sigma))) * (b1 * sin(w1/sigma) - a1 * cos(w1/sigma))

    # b1
    bplus[0] = -2 * (e * (-(g2/sigma))) * cos(w2/sigma) -2 * (e * (-(g1/sigma))) * cos(w1/sigma)

    # b2
    bplus[1] = 4 * cos(w2/sigma) * cos(w1/sigma) * (e ** (-(g1 +g2)/sigma))\
            + (e * (-2 * (g2/sigma))) + (e * (-2 * (g1/sigma)))

    # b3
    bplus[2] = -2 * cos(w1/sigma) * (e ** (-(g1 + 2*g2)/sigma)) - 2 *\
            cos(w2/sigma) * (e ** (-(g2 + 2*g1)/sigma))

    # b4
    bplus[3] = (e ** (-(2*g1 + 2*g2)/sigma))

    return aplus, bplus


def anti_causal_coeff(aplus, bplus):
    aminus = np.zeros(shape=(4,))
    aminus[0] = aplus[1] - bplus[0] * aplus[0]
    aminus[1] = aplus[2] - bplus[1] * aplus[0]
    aminus[2] = aplus[3] - bplus[2] * aplus[0]
    aminus[3] = - bplus[3] * aplus[0]

    return aminus, bplus[:]


def anticausal_filter(n, mask, x, index, a_minus, b_minus):
    anticausal_output = 0

    #base condition
    if n > mask.shape[1] - 5:
        mask[index:index+1, n] = anticausal_output
        return anticausal_output

    # first sum
    mul = a_minus * x[n: n + 4]

    # second sum
    sum2 = 0
    for m, bm in enumerate(b_minus, 1):

        if mask[index:index+1, n] < 0:
            sum2 += bm * anticausal_filter(n + m, mask, x, index,a_minus, b_minus )
        else:
            sum2 += bm * mask[index:index+1, n]

    anticausal_output = mul.sum() - sum2

    mask[index:index+1, n] = anticausal_output
    return anticausal_output


def causal_filter(n, mask, x, index, a_plus, b_plus):
    causal_output = 0

    if n < 3:
        mask[index:index+1, n] = causal_output
        return causal_output

    # first sum
    mul = x[n-3: n+1] * a_plus

    # second sum
    sum2 = 0
    for m, bm in enumerate(b_plus, 1):

        if mask[index:index+1, n] < 0:
            sum2 += bm * causal_filter(n-m, mask, x, index, a_plus,b_plus )
        else:
            sum2 += bm * mask[index:index+1, n]

    causal_output = mul.sum() - sum2

    mask[index:index+1, n] = causal_output
    return causal_output

def recursive_filter(image, sigma):
    # create coefficients
    a_plus, b_plus = causal_coeff(sigma)
    a_minus, b_minus = anti_causal_coeff(a_plus, b_plus)
    a_plus = a_plus[::-1]

    n = image.shape[1]
    # masks creation
    causal_mask = image[:] * -1
    anticausal_mask = image[:] * -1

    # apply causal filter
    for i, row in enumerate(image):
        causal_filter(n-1, causal_mask, row, i, a_plus, b_plus)

    # apply anti-causal filter
    for i, row in enumerate(image):
        anticausal_filter(1, anticausal_mask, row, i, a_minus, b_minus)

    response = (1.0/sigma * sqrt(2.0 * pi)) * (causal_mask + anticausal_mask)

    msc.imsave("task3_causal.png", causal_mask)
    msc.imsave("task3_anticausal.png", anticausal_mask)
    msc.imsave("task3_result.png", abs(response))

    return(abs(response))

def task3(image):
    size = [0, 1, 3, 4, 5, 6]

    sigma_075 = np.zeros(len(size))
    sigma_2 = np.zeros(len(size))
    sigma_375 = np.zeros(len(size))
   
    # sigma 0.752
    for i in range(len(size)):
        start = time.time()
        recursive_filter(image, sigma=0.75)
        stop = time.time()
        sigma_075[i] = stop - start

   # sigma 1.752
    for i in range(len(size)):
        start = time.time()
        recursive_filter(image, sigma=2.0)
        stop = time.time()
        sigma_2[i] = stop - start

   #sigma 3.752
    for i in range(len(size)):
        start = time.time()
        recursive_filter(image, sigma=3.75)
        stop = time.time()
        sigma_375[i] = stop - start

    fig = plt.figure()

    ax = fig.add_axes([0.1, 0.1, 0.6, 0.75])
    ax.plot(size, np.log(sigma_075*1000), 'k--', lw=2, label='sig 0.75')
    ax.plot(size, np.log(sigma_2*1000), 'k:', lw=2, label='sig 2.0')
    ax.plot(size, np.log(sigma_375*1000), 'r--', lw=2, label='sig 3.75')

    ax.set_xlabel('Sigma', size=25)
    ax.set_ylabel('Log(time*1000)', size=25)
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig("task3plot.png")
    plt.show()



def main():

    face = readIntensityImage("bauckhage.jpg")
   # clock = msc.imread("images/clock.jpg").astype("float")
    task3(face)
#    task3(clock)

main()
