import numpy as np
from scipy import signal
from scipy import misc
from scipy.misc import imread
import matplotlib.pyplot as plt

def findc2(sig, theta, size):
    mid=size/2.0
    ret = 0
    sum1=0
    for x in range(size):
        for y in range(size):
            dot = np.cos(theta)*(x-mid)+np.sin(theta)*(y-mid)
            firstEx = np.exp(1j * (np.pi/(2*sig))*dot)
            secondEx = np.exp(-((x-mid)**2+(y-mid)**2)/(2*sig*sig))
            sum1+=firstEx*secondEx
    sum2=0
    for x in range(size):
        for y in range(size):
            secondEx = np.exp(-((x-mid)**2+(y-mid)**2)/(2*sig*sig))
            sum2+=secondEx
    ret=sum1/sum2
    return ret.real


def findc1(sig, theta, size, c2):
    sum1=0
    mid = size/2.0
    for x in range(size):
        for y in range(size):
            dot = np.cos(theta)*(x-mid)+np.sin(theta)*(y-mid)
            firstEx = np.exp(1j * (np.pi/(2*sig))*dot)
            firstExC = np.exp(-1j * (np.pi/(2*sig))*dot)
            secondEx = np.exp(-((x-mid)**2+(y-mid)**2)/(2*sig*sig))
            sum1+=(firstEx-c2)*secondEx*(firstExC-c2)*secondEx
    root = np.sqrt(sum1.real)
    
    ret=1/root
    ret=ret*sig*sig
    return ret.real


def twoMorlet(sig, theta, c1, c2, x,y,size):
    mid = size/2.0
    dot = np.cos(theta)*(x-mid)+np.sin(theta)*(y-mid)
    firstEx = np.exp(1j * (np.pi/(2*sig))*dot)
    secondEx = np.exp(-((x-mid)**2+(y-mid)**2)/(2*sig*sig))
    return (c1/sig)*(firstEx-c2)*secondEx


def wavelet(sig, theta, size):
    image=[ [0 for x in range(size)] for y in range(size)]
    c2=findc2(sig,theta,size)
    c1=findc1(sig,theta,size,c2)
    for x in range(size):
        for y in range(size):
            image[x][y]=twoMorlet(sig,theta,c1,c2,x,y,size)
    return image



wave = np.array(wavelet(6,0,37))
plt.imshow(wave.real)
plt.show()
plt.imshow(wave.imag)
plt.show()


LENA = misc.lena()
plt.imshow(LENA)
plt.show()
convolution = signal.convolve2d(LENA, wave.real, 'same')
plt.imshow(convolution)
plt.show()



