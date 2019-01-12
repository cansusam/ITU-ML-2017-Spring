import matplotlib.pyplot as plt
import numpy as np
import math
from itertools import groupby
import matplotlib.cm as cm
from sklearn.feature_selection import VarianceThreshold
from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))

np.set_printoptions(threshold=10)

classNumber = 10
featureNumber = 64

def euclideandistanceCalc(point,center):
    squareSums = 0.0
    for p_i, c_i in zip(point,center):
        squareSums += (p_i-c_i)**2
    return np.sqrt(squareSums)

trainingDataSet = np.loadtxt(d+'/data/optdigits.tra', delimiter=',');
testDataSet = np.loadtxt(d+'/data/optdigits.tes', delimiter=',');

trainingDataNumber = trainingDataSet.shape[0]
testDataNumber = testDataSet.shape[0]

# store labels of each sample
trainingLabels = trainingDataSet[:, 64]
testLabels = testDataSet[:, 64]

# remove lables from features
trainingDataSet = trainingDataSet[:,:64]
testDataSet = testDataSet[:,:64]

K = 10 # number of k
centers = np.asarray([np.random.randint(0, 16, 64).tolist() for i in range (0,K)])
preCenters = np.copy(centers)
#print(randomCenters)
sampleCenterRelation = np.zeros((trainingDataNumber, K))

loopContinue = True

iterationCounter = 0

while(loopContinue):
    iterationCounter += 1
    # find minimum distance center for each sample
    for t in range(0,trainingDataNumber):
        x_t = trainingDataSet[t,:]
        minIndex = -1
        minValue = float('inf')
        for i in range(0,K):
            c_i = centers[i]
            if euclideandistanceCalc(x_t,c_i) < minValue :
                minIndex = i
                minValue = euclideandistanceCalc(x_t,c_i)
        result = np.zeros([1,K])
        result[0,minIndex] = 1
        sampleCenterRelation[t] = result
        #print(sampleCenterRelation)
        #print(minValue)
    # recalculate each center according to new relations
    for i in range(0,K):
        indices = np.where(sampleCenterRelation[:,i] == 1)[0]
        # if indices.shape[0] == 0, that center does not include any member
        centers[i] = trainingDataSet[indices,:].sum(axis=0)/indices.shape[0]
    # if the values are the same with previous iteration, exit
    if(centers == preCenters).all():
        loopContinue = False
    else:
        preCenters = np.copy(centers)

print("Centers stabilized after ", iterationCounter, " iterations")

# Reconstruction error calculation
error = 0.0
for t in range(0,trainingDataNumber):
    x_t = trainingDataSet[t,:]
    for i in range(0,K):
        c_i = centers[i]
        if sampleCenterRelation[t,i] == 1:
            error += euclideandistanceCalc(x_t,c_i)**2
print(error)

# Gini Index Calculation

giniError = np.zeros([K])
for i in range(0, K):
    indices = np.where(sampleCenterRelation[:, i] == 1)[0]
    samplesInRegion = trainingDataSet[indices, :]
    labelsInRegion = trainingLabels[indices]
    if indices.shape[0] != 0:
        for k in range(0, classNumber):
            purity = np.where(labelsInRegion[:] == k)[0].shape[0] / indices.shape[0]
            giniError[i] += purity * (1 - purity)

averageGini = np.average(giniError)
print(giniError)
print(averageGini)