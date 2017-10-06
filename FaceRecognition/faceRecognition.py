import scipy.io as sio
import numpy as np
from scipy import signal
from scipy import misc
import matplotlib.pyplot as plt
import math
import cv2
from skimage import measure

from PIL import Image
import os, sys

from scipy import ndimage

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

def block_mean(ar, fact):
    assert isinstance(fact, int), type(fact)
    sx, sy = ar.shape
    X, Y = np.ogrid[0:sx, 0:sy]
    regions = sy/fact * (X/fact) + Y/fact
    res = ndimage.mean(ar, labels=regions, index=np.arange(regions.max() + 1))
    res.shape = (sx/fact, sy/fact)
    return res


def median(lst):
    return np.median(np.array(lst))

def mean(lst):
    return np.mean(np.array(lst))

sigma = [4,8]
theta = [0, math.pi/4, math.pi/2, math.pi*3/4]

def flat(lst):
     flatlist = []
     for level1 in lst:
         templist = []
         for level2 in level1:
             templist.append(level2.ravel())
         flatlist.append(np.asarray(templist).ravel())
     return(np.asarray(flatlist));

def downsample(list, pic):
    for s in sigma:
        for t in theta:
            wave = np.array(wavelet(s,t,37))
            convolutionReal = signal.convolve2d(pic, wave.real, 'same')
            convolutionImag = signal.convolve2d(pic, wave.imag, 'same')
            mag = np.sqrt(np.power(convolutionImag,2)+np.power(convolutionReal,2))
            blurLayer1 = cv2.GaussianBlur(mag,(5,5),8)
            #downsample
            blurLayer1.resize((4,4),refcheck=False)
            list.append(blurLayer1)
    return(list)
#
#
path = ('/Users/Yougen/Desktop/Computer Vision/untitled/')


def start():
    allVector = []
    numberOfFace = 0;
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.endswith(("pgm")):
                numberOfFace = numberOfFace + 1;
                pic = cv2.imread(os.path.join(root, name),0)
                pic.resize((96,96),refcheck=False)
                blur = cv2.GaussianBlur(pic,(5,5),8)
                blur.resize((4,4),refcheck=False)
                listBlur = []
                listBlur.append(blur)
                allVector.append((downsample(listBlur,pic)))

    averageFace = []
    for x in range(len(allVector[0])):
         sum = np.zeros((4, 4))
         for y in range(len(allVector)):
             sum = sum + allVector[y][x]
         averageFace.append(sum/numberOfFace)


    difference = []
    allDifference = []
    for image in allVector:
        vectorDifference = []
        euclidDistance = 0;
        for level in range(len(image)):
            for x in range(len(image[level])):
                for y in range(len(image[level][0])):
                    euclidDistance = euclidDistance + (math.pow((image[level][x][y] - averageFace[level][x][y]),2))
                    vectorDifference.append(image[level][x][y] - averageFace[level][x][y])
        euclidDistance = math.sqrt(euclidDistance)
        difference.append(euclidDistance)
        allDifference.append(np.asarray(vectorDifference))

    allDifference = np.asarray(allDifference)
    sumOfDistance = 0;
    print("median: ")
    print(median(difference))
    print("mean: ")
    print(mean(difference))
    print("face: ")
    print(numberOfFace)

    covariance = np.transpose(allDifference).dot(allDifference)/(numberOfFace-1)
    (eigenVector, eigenValue, eigenVectorTrans) = np.linalg.svd(covariance)
    print("covariance shape:")
    print(covariance.shape)
    print("eigenValue shape:")
    print(eigenValue.shape)
    print("eigenVector shape:")
    print(eigenVector.shape)
    print("eigenVector[0] shape:")
    print(eigenVector[0].shape)
    Y = allDifference.dot(eigenVector)
    print("allDifference shape:")
    print(allDifference.shape)
    print("Y shape:")
    print(Y.shape)
    print("eigenValue is:")
    print(eigenValue)

    reduce = 0
    for reduceIndex,eig in enumerate(eigenValue):
        if eig < 0.25:
            reduce = reduceIndex
            break;
    reducedEigenVector = eigenVector[:,0:reduce-1]

    reducedY = allDifference.dot(reducedEigenVector)


start()