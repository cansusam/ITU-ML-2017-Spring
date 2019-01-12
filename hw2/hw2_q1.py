# Lecture:  BLG 527E Machine Learning
# Term:     2018 - Spring
# Student:  Omercan Susam
# ID:       504162517

import numpy as np
import math
from enum import Enum
from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__))) # parent directory

class method(Enum):
    euclideanDistance = 0
    weightedDistance = 1

np.set_printoptions(threshold=100,precision=2)

classNumber = 10
featureNumber = 64

theFile = open('results_hw2_q1.txt', 'w')

def euclideanDistanceCalc(point, center):
    squareSums = 0.0
    for p_i, c_i in zip(point, center):
        squareSums += (p_i - c_i) ** 2
    return math.sqrt(squareSums)


'''
    features can be considered as pixels separated over 8x8 matrix
    then it can be transformed to array
'''


def weightedDistanceCalc(point, center):
    size = int(math.sqrt(featureNumber))
    sizediv2Ceil = math.ceil(size / 2)
    sizediv2Floor = math.floor(size / 2)
    mat = np.zeros([size, size])
    maxV = 2
    minV = 1
    incV = (maxV - minV) / (sizediv2Floor - 1)
    for i in range(0, sizediv2Floor):
        for j in range(0, i + 1):
            mat[i, j] = minV + j * incV  # triangle (upper left bottom)
            mat[j, i] = mat[i, j]  #  first square (upper left)
            # mat[i,size-(j+1)] = mat[i,j] # second triangle (upper right bottom)
            # mat[j,size-(i+1)] = mat[i,j] # second square (upper right)

    # mirror matrix vertical and horizontal
    mat = mat + np.flip(mat, 0)
    mat = mat + np.flip(mat, 1)

    # 2d to 1d
    mat = mat.flatten()
    squareSums = 0.0
    for m_i, p_i, c_i in zip(mat, point, center):
        squareSums += (m_i*(p_i - c_i)) ** 2

    # print(mat)
    return math.sqrt(squareSums)


trainingDataSet = np.loadtxt(d+'/data/optdigits.tra', delimiter=',');
testDataSet = np.loadtxt(d+'/data/optdigits.tes', delimiter=',');

trainingDataNumber = trainingDataSet.shape[0]
testDataNumber = testDataSet.shape[0]

# store labels of each sample
trainingLabels = trainingDataSet[:, 64]
testLabels = testDataSet[:, 64]

#  remove lables from features
trainingDataSet = trainingDataSet[:, :64]
testDataSet = testDataSet[:, :64]

Ks = [10, 20, 30]  #  number of k
for K in Ks:
    centers = np.asarray([np.random.randint(0, 16, 64).tolist() for i in range(0, K)])
    #  one copy to remember previous states of the random points
    preCenters = np.copy(centers)
    #  one copy to use same initial random points for comparison with weighted distance
    centersCopy = np.copy(centers)
    # print(randomCenters)
    sampleCenterRelation = np.zeros((trainingDataNumber, K))

    #  0 for euclidean, 1 for weighted
    for distanceMethod in range(0, 2):
        print("K = ", K, " ", method(distanceMethod))

        theFile.write("K = %d %s\n" % (K,method(distanceMethod)))
        loopContinue = True
        iterationCounter = 0
        if distanceMethod == 1:
            centers = np.copy(centersCopy)
        while (loopContinue):
            iterationCounter += 1
            # find minimum distance center for each sample
            for t in range(0, trainingDataNumber):
                x_t = trainingDataSet[t, :]
                minIndex = -1
                minValue = float('inf')
                for i in range(0, K):
                    c_i = centers[i]
                    if distanceMethod == 0:
                        if euclideanDistanceCalc(x_t, c_i) < minValue:
                            minIndex = i
                            minValue = euclideanDistanceCalc(x_t, c_i)
                    elif distanceMethod == 1:
                        if weightedDistanceCalc(x_t, c_i) < minValue:
                            minIndex = i
                            minValue = weightedDistanceCalc(x_t, c_i)
                result = np.zeros([1, K])
                result[0, minIndex] = 1
                sampleCenterRelation[t] = result
                # print(sampleCenterRelation)
                # print(minValue)
            # recalculate each center according to new relations
            for i in range(0, K):
                indices = np.where(sampleCenterRelation[:, i] == 1)[0]
                if indices.shape[0] != 0:
                    centers[i] = trainingDataSet[indices, :].sum(axis=0) / indices.shape[0]
            # if the values are the same with previous iteration, exit
            if (centers == preCenters).all():
                loopContinue = False
            else:
                preCenters = np.copy(centers)

        print("Centers stabilized after ", iterationCounter, " iterations")
        theFile.write("Centers stabilized after %d iterations \n"% iterationCounter)
        #  Reconstruction error calculation
        error = np.zeros([K])
        for t in range(0, trainingDataNumber):
            x_t = trainingDataSet[t, :]
            for i in range(0, K):
                c_i = centers[i]
                if sampleCenterRelation[t, i] == 1:
                    error[i] += euclideanDistanceCalc(x_t, c_i) ** 2
                    '''
                    if distanceMethod == 0:
                        error[i] += euclideanDistanceCalc(x_t, c_i) ** 2
                    elif distanceMethod == 1:
                        error[i] += weightedDistanceCalc(x_t, c_i) ** 2
                    '''
        averageError = np.average(error)
        print("Error for clusters: \n", error)
        theFile.write("Error for clusters: \n")
        np.savetxt(theFile, error,fmt='%1.3f')
        print("Average Error: ", averageError)
        theFile.write("\nAverage Error: %f \n"% averageError)

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
        print("Gini Impurities for clusters: \n",giniError)
        theFile.write("Gini Impurities for clusters: \n")
        np.savetxt(theFile,giniError,fmt='%1.3f')
        print("Average Gini Impurity: ",averageGini, "\n")
        theFile.write("Average Gini Impurity: %f"%averageGini)
        theFile.write("\n")
