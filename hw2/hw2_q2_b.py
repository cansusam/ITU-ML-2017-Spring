# Lecture:  BLG 527E Machine Learning
# Term:     2018 - Spring
# Student:  Omercan Susam
# ID:       504162517

import numpy as np
import math
from enum import Enum
import matplotlib.pyplot as plt
from os.path import dirname, abspath
d = dirname(dirname(abspath(__file__)))

class method(Enum):
    euclideanDistance = 0
    weightedDistance = 1

np.set_printoptions(threshold=100,precision=2)

classNumber = 10
featureNumber = 64

theFile = open('results_hw2_q2_b.txt', 'w')

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

def findMajorityLabel(sampleCenterRelation,trainingLabels,K):
    finalClusterLabels = np.full(K,-1) # -1 given to find clusters with zero members
    for i in range(K):
        indices = np.where(sampleCenterRelation[:, i] == 1)[0]
        if indices.shape[0] != 0:
            currentClusterLabels = trainingLabels[indices]
            labels , freq = np.unique(currentClusterLabels, return_counts=True)
            finalClusterLabels[i] = labels[np.where(freq == freq.max())]
    return finalClusterLabels


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

# ============================================
#   Cluster Calculation based on training data
# ============================================
K=30 # cluster number
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
        #print("\r%d" % iterationCounter, end=" ", flush=True)
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
    # ============================================
    #   Find closest cluster center
    # ============================================
    testLabelResults = np.full(testDataNumber,-1)
    clusterLabels = findMajorityLabel(sampleCenterRelation,trainingLabels,K)
    for t in range(0,testDataNumber):
        x_t = testDataSet[t, :]
        minIndex = -1
        minValue = float('inf')
        for i in range(0,K):
            # if cluster does not have zero members (or it has a label)
            if(clusterLabels[i] != -1):
                c_i = centers[i]
                if distanceMethod == 0:
                    if euclideanDistanceCalc(x_t, c_i) < minValue:
                        minIndex = i
                        minValue = euclideanDistanceCalc(x_t, c_i)
                elif distanceMethod == 1:
                    if weightedDistanceCalc(x_t, c_i) < minValue:
                        minIndex = i
                        minValue = weightedDistanceCalc(x_t, c_i)
        testLabelResults[t] = clusterLabels[minIndex]

    freq = np.unique(testLabels, return_counts=True)[1]
    matchingClassNumbers = np.zeros(classNumber)
    freqResults = np.zeros(classNumber)
    for x in range(0, classNumber):
        matchingClassIndex = (testLabelResults == x) & (testLabels == x)
        matchingClassNumbers[x] = sum(matchingClassIndex)
        freqResults[x] = matchingClassNumbers[x] / freq[x]

    # find results for accuracy
    print("\nClass\t", "Accuracy\t")
    theFile.write("Class\tAccuracy\t\n")
    freqMatrix = np.zeros((classNumber, classNumber))
    for x in range(0, classNumber):
        subsetOfMust = testLabelResults[testLabels == x]
        for y in range(0, classNumber):
            freqMatrix[x, y] = sum(subsetOfMust == y)
        print(x, "\t", float("%0.3f" % (freqResults[x] * 100)), "%\t")
        theFile.write("%d\t" % x)
        theFile.write("%0.3f%%\t\n" % (freqResults[x] * 100))

    # confusion matrix calculation
    confusionMatrix = np.zeros((classNumber, classNumber));
    for x in range(0, testDataNumber):
        prediction = int(testLabelResults[x]);
        actual = int(testLabels[x]);
        confusionMatrix[actual, prediction] = confusionMatrix[actual, prediction] + 1 / freq[actual];

    # plot confusion matrix
    fig, ax1 = plt.subplots(1, 1)
    ax1.imshow(confusionMatrix, cmap='Blues', alpha=.9, interpolation='nearest')
    ax1.set_xticks(np.arange(0, 10, 1))
    ax1.set_yticks(np.arange(0, 10, 1))
    fig.suptitle("Confusion Matrix")
    fig.savefig("hw2_q2_b_k_%d_dis_%d" % (K, distanceMethod))
    # plt.show()
