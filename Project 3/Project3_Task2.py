import numpy as np
import scipy.misc as msc
import scipy.ndimage as ndimage

def warping_y(img, amp, ph,freq): #to check effect of different parameters(amp,phase, frequency)
    rows, cols = img.shape
    x,y = np.meshgrid(np.linspace(0,cols,cols), np.linspace(-amp, rows+amp, rows+(2*amp)))   
    ys = y + amp*np.sin(2*np.pi*(x/rows)*freq + ph)
    xs,ys = x.flatten(), ys.flatten()
    X = np.vstack((ys,xs))
    output_image = ndimage.interpolation.map_coordinates(img,X).reshape(rows+2*amp, cols)
    return output_image

def warping_xy(img, amp, ph, frq):
    ampx, ampy = amp[0], amp[1]
    frqx, frqy = frq[0], frq[1]
    phx, phy = ph[0], ph[1]
    rows, cols = img.shape
    x,y = np.meshgrid(np.linspace(-ampx,cols+ampx,cols+(2*ampx)), np.linspace(-ampy,rows+ampy,rows+(2*ampy)))    
    xs = x + ampx * np.sin(2*np.pi*(y/cols) * frqx + phx)
    ys = y + ampy * np.sin(2*np.pi*(x/rows) * frqy + phy)
    xs,ys = xs.flatten(), ys.flatten()
    X = np.vstack((ys,xs)) 
    output_image = ndimage.interpolation.map_coordinates(img,X).reshape(rows+2*ampy, cols+2*ampx)
    return output_image

clock = msc.imread("images/clock.jpg", flatten=True).astype('float')
ph = 0

#first output
result1 = warping_y(clock, 40, ph, 0.5)
msc.imsave("result_images/warps/output1.png", result1)

#second output
result2 = warping_y(clock, 100, ph, 1.0)
msc.imsave("result_images/warps/output2.png", result2)

#third output
result3 = warping_xy(clock, [8, 10], [0, -np.pi/2], [2.0, 2.0])
msc.imsave("result_images/warps/output3.png", result3)

#fourth output
res4 = warping_xy(clock, [0, 10], [0, 0], [0, 4.0])
#msc.imsave("resulting_images/warps/face.png", res4)
result4 = warping_xy(res4, [15, 0], [-np.pi/2, 0], [6.0, 0])
msc.imsave("result_images/warps/output4.png", result4)

#fifth output
res5 = warping_xy(clock, [0, 10], [0, 0], [0, 8.0])
#msc.imsave("resulting_images/warps/face.png", res5)
result5 = warping_xy(res5, [12, 0], [np.pi, 0], [1.0, 0])
msc.imsave("result_images/warps/output5.png", result5)

# Generate images with different amplitudes
amplitudes = [0, 5, 10, 20, 50, 100]
for amp in amplitudes:
    clock_warp = warping_y(clock, amp, ph, 2.0)
    msc.imsave("result_images/warps/amplitude/clock_amp_{}.png".format(amp), clock_warp)
  
# Generate images with different phases
phases = [-np.pi/2, 0, np.pi/2, np.pi]
for ph in phases:
    clock_warp = warping_y(clock, 10, ph, 2)
    msc.imsave("result_images/warps/phase/clock_phase_{}.png".format(ph), clock_warp)

# Generate images with different frequencies
freqs = [1, 2, 4, 10, 20, 5]
for frq in freqs:
    clock_warp = warping_y(clock, 10, ph, frq)
    msc.imsave("result_images/warps/frequency/clockface_freq_{}.png".format(frq), clock_warp)
