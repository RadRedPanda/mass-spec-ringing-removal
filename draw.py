import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

from math import log
from scipy.cluster.hierarchy import dendrogram, linkage
import numpy as np
from numpy import random

from numpy import genfromtxt

import matplotlib.cm as cm

import glob
import pandas as pd

inputFile = ['0_original exported-Sheet1.csv']
outputFile = ['0_output.csv']
path = 'DataForKevin/slice/data/*'

error = 0.003
cutoff = 1000000
noiseCut = 100
allowance = 1.5
avgRate = 1.5

def outlier(curr, nex, pre):
    if abs(pre / nex - avgRate) < abs(curr / nex - avgRate):
        return -2
    else:
        return -1

def ring(i, arrx, arry):
    currX = arrx[i]
    maxX = log(arry[i]) / 800
    arrd = []
    temp = i - 2
    arrd.append(temp + 1)
    while (temp >= 0 and arrx[i] - arrx[temp] < maxX ) :
        arrd.append(temp)
        # weird data
        if (arry[temp] > arry[arrd[-2]] * allowance) :
            arrd.pop(outlier(arry[temp], arry[temp - 1], arry[arrd[-2]]))
        temp-=1
    temp = i + 2
    arrd.append(temp - 1)
    while (temp < len(arry) and arrx[i] - arrx[temp] > -maxX) :
        arrd.append(temp)
        if (arry[temp] > arry[arrd[-2]] * allowance) :
            arrd.pop(outlier(arry[temp], arry[temp + 1], arry[arrd[-2]]))
        temp+=1
    arrd.sort(reverse = True)
    for x in arrd:
        arry[i] += arry[x]
    return np.delete(arrx, arrd), np.delete(arry, arrd), int(np.argwhere(arrx==currX)[0])

name = []
prev = []
post = []
    
def cutDown(ifn, ofn):
    if ifn[-3:] == 'csv':
        return 0, 0
    print ifn,
    name.append(ifn)
    print ' \t',
    ogData = pd.read_excel(ifn, skiprows = 7, convert_float = False).values
    #print(newData.values)
    #ogData = genfromtxt(ifn, skip_header = 8, encoding = None, delimiter = ',', dtype = float)
    a = ogData[:,0]
    b = ogData[:,1]
    x = 0
    while x < len(a):
        if b[x] < noiseCut:
            b = np.delete(b, x)
            a = np.delete(a, x)
        else:
            x+=1
    x = 0
    print len(b),
    prev.append(len(b))
    while x < len(b):
        if b[x] > cutoff:
            a, b, x = ring(x, a, b)
        x+=1
    print(len(b))
    post.append(len(b))
    newArray = np.column_stack((a, b))
    np.savetxt(ofn, newArray, delimiter=',')
    return a, b

fig, ax1 = plt.subplots()
files = glob.glob(path)
for f in files:
    xAxis, yAxis = cutDown(f, f[:-5] + ' out.csv')

newerArray = np.column_stack((name, prev, post))
np.savetxt('results.csv', newerArray, delimiter=',', fmt = '%s')

bars = ax1.bar(xAxis, yAxis, 0.001)
for bar in range(len(bars)):
    bars[bar].set_color('b')
plt.show()
