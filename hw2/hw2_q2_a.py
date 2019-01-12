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

theFile = open('results_hw2_q2_a.txt', 'w')

Ks = [1,5]  #  number of k
for K in Ks:
    #  0 for euclidean, 1 for weighted
    for distanceMethod in range(0, 2):
        print("\nK = ", K, " ", method(distanceMethod))
        theFile.write("K = %d %s\n" % (K,method(distanceMethod)))
        # initialize prediction labels as "-1" since it does not included in original labels, easier to find errors
        predictedTestLabels = np.full(testDataNumber,-1)
        for i in range(0,testDataNumber):
            #print("\r%d" %i,end=" ", flush=True)
            test_i = testDataSet[i,:]
            # minimum k neighbors with their indexes kept in this array
            minDistanceK = np.full([K,2],(float('inf')))
            for j in range(0,trainingDataNumber):
                train_j = trainingDataSet[j,:]
                if distanceMethod == 0:
                    distance = euclideanDistanceCalc(test_i, train_j)
                    # if the distance is shorter than any of the elements in the list
                    if (distance < minDistanceK[:,0]).any():
                        # TODO find max valued distance, replace it with new one, sort the array
                        # keeping array sorted may reduce computation time
                        # f[np.argsort(f[:,1])] sort according to column 1
                        indexes = np.where(minDistanceK[:,0] > distance)[0]
                        # replace with the one it has lower index (to keep sorted)
                        minDistanceK[indexes[0]] = [distance, j]
                elif distanceMethod == 1:
                    distance = weightedDistanceCalc(test_i, train_j)
                    if (distance < minDistanceK[:,0]).any():
                        indexes = np.where(minDistanceK[:,0] > distance)[0]
                        minDistanceK[indexes[0]] = [distance, j]
            # classify test data
            labelOccurances = np.zeros(classNumber)
            for p in range(K):
                labelOccurances[int(trainingLabels[int(minDistanceK[p,1])])] += 1
            maxIndexes = np.where(labelOccurances == labelOccurances.max())[0]
            #print(labelOccurances)
            #print(maxIndexes)
            # if many indices occurred in same number, select randomly
            if len(maxIndexes) > 1:
                predictedTestLabels[i] = maxIndexes[np.random.randint(0,len(maxIndexes))]
            else:
                predictedTestLabels[i] = maxIndexes[0]

        # class frequencies

        freq = np.unique(testLabels, return_counts=True)[1]
        matchingClassNumbers = np.zeros(classNumber)
        freqResults = np.zeros(classNumber)
        for x in range(0, classNumber):
            matchingClassIndex = (predictedTestLabels == x) & (testLabels == x)
            matchingClassNumbers[x] = sum(matchingClassIndex)
            freqResults[x] = matchingClassNumbers[x] / freq[x]

        # find results for accuracy
        print("\nClass\t", "Accuracy\t")
        theFile.write("Class\tAccuracy\t\n")
        freqMatrix = np.zeros((classNumber, classNumber))
        for x in range(0, classNumber):
            subsetOfMust = predictedTestLabels[testLabels == x]
            for y in range(0, classNumber):
                freqMatrix[x, y] = sum(subsetOfMust == y)
            print(x, "\t", float("%0.3f" % (freqResults[x] * 100)), "%\t")
            theFile.write("%d\t" % x)
            theFile.write("%0.3f%%\t\n" % (freqResults[x] * 100))

        # confusion matrix calculation
        confusionMatrix = np.zeros((classNumber, classNumber));
        for x in range(0, testDataNumber):
            prediction = int(predictedTestLabels[x]);
            actual = int(testLabels[x]);
            confusionMatrix[actual, prediction] = confusionMatrix[actual, prediction] + 1 / freq[actual];

        # plot confusion matrix
        fig, ax1 = plt.subplots(1, 1)
        ax1.imshow(confusionMatrix, cmap='Blues', alpha=.9, interpolation='nearest')
        ax1.set_xticks(np.arange(0, 10, 1))
        ax1.set_yticks(np.arange(0, 10, 1))
        fig.suptitle("Confusion Matrix")
        fig.savefig("hw2_q2_a_k_%d_dis_%d" %(K,distanceMethod))
        #plt.show()