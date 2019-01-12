# Lecture:  BLG 527E Machine Learning
# Term:     2018 - Spring
# Student:  Omercan Susam
# ID:       504162517

import numpy as np
import pomegranate
from os.path import dirname, abspath
import os
import time


cwd = os.getcwd()
timestr = time.strftime("%Y%m%d-%H%M%S")
d = dirname(dirname(abspath(__file__)))
# write results
the_file = open('hw3/results-HMM-'+ timestr +'-.txt', 'w');

classNumber = 2
featureNumber = 64

trainingDataSet = np.loadtxt(d+'/data/optdigits.tra', delimiter=',');
testDataSet = np.loadtxt(d+'/data/optdigits.tes', delimiter=',');

# store labels of each sample
trainingLabels = trainingDataSet[:, 64]
testLabels = testDataSet[:, 64]

# update labels
trainingLabels[trainingLabels < 6] = 0
trainingLabels[trainingLabels >= 6] = 1

testLabels[testLabels < 6] = 0
testLabels[testLabels >= 6] = 1

traininIndexes1 = np.where(trainingLabels == 1)
traininIndexes0 = np.where(trainingLabels == 0)

testIndexes1 = np.where(testLabels == 1)
testIndexes0 = np.where(testLabels == 0)

#  remove lables from features
trainingDataSet = trainingDataSet[:, :64]
testDataSet = testDataSet[:, :64]

trainingDataSet1 = trainingDataSet[traininIndexes1,:]
trainingDataSet0 = trainingDataSet[traininIndexes0,:]

testDataSet1 = trainingDataSet[testIndexes1,:]
testDataSet0 = trainingDataSet[testIndexes0,:]

shapeTra = trainingDataSet.shape[0]
shapeTra0 = trainingDataSet0[0].shape[0]
shapeTra1 = trainingDataSet1[0].shape[0]
shapeTest = testDataSet.shape[0]

modelFor0 = pomegranate.HiddenMarkovModel.from_samples(pomegranate.DiscreteDistribution, X=trainingDataSet0[0], n_components=2)
modelFor1 = pomegranate.HiddenMarkovModel.from_samples(pomegranate.DiscreteDistribution, X=trainingDataSet1[0], n_components=2)
resultsTraining = np.zeros(shapeTra)
resultsTest = np.zeros(shapeTest)

for i in range(0, trainingDataSet.shape[0]):
    predict0 = modelFor0.log_probability(trainingDataSet[i])
    predict1 = modelFor1.log_probability(trainingDataSet[i])
    if predict0 > predict1:
        resultsTraining[i] = 0
    else:
        resultsTraining[i] = 1

for i in range(0, testDataSet.shape[0]):
    predict0 = modelFor0.log_probability(testDataSet[i])
    predict1 = modelFor1.log_probability(testDataSet[i])
    if predict0 > predict1:
        resultsTest[i] = 0
    else:
        resultsTest[i] = 1


the_file.write("\nClass\tAccuracy\tTraining\n")
for i in range(0, classNumber):
    matchingTrainingNum = np.count_nonzero((resultsTraining == i) & (trainingLabels == i))
    actualTrainingNum = np.count_nonzero(trainingLabels == i)
    print(i, "\t", float("%0.3f" % (matchingTrainingNum/actualTrainingNum * 100)), "%\t")
    the_file.write("%d\t" % i)
    the_file.write("%0.3f%%\t\n" % (matchingTrainingNum/actualTrainingNum * 100))


the_file.write("\nClass\tAccuracy\tTest\n")
for i in range(0, classNumber):
    matchingTestNum = np.count_nonzero((resultsTest == i) & (testLabels == i))
    actualTestNum = np.count_nonzero(testLabels == i)
    print(i, "\t", float("%0.3f" % (matchingTestNum/actualTestNum * 100)), "%\t")
    the_file.write("%0.3f%%\t\n" % (matchingTestNum/actualTestNum* 100))
    the_file.write("%d\t" % i)


print("")