# Lecture:  BLG 527E Machine Learning
# Term:     2018 - Spring
# Student:  Omercan Susam
# ID:       504162517

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
from hmmlearn.hmm import MultinomialHMM
from os.path import dirname, abspath
import os
import time


cwd = os.getcwd()
timestr = time.strftime("%Y%m%d-%H%M%S")
d = dirname(dirname(abspath(__file__)))
# write results
the_file = open('hw3/results-'+ timestr +'-.txt', 'w');

classNumber = 10
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

def createStreams(data):
    stream = data[0][0]
    lengths = []
    lengths.append(data[0].shape[1])
    for i in range(1, data[0].shape[0]):
        stream = np.concatenate([stream, data[0][i]])
        lengths.append(data[0].shape[1])
    # lengths = np.full((data[0].shape[0],1),data[0].shape[1])
    return stream.reshape(stream.shape[0],1), lengths

streamTrain0, lengthsTrain0 = createStreams(trainingDataSet0)
streamTrain1, lengthsTrain1 = createStreams(trainingDataSet1)
streamTest0, lengthsTest0 = createStreams(testDataSet0)
streamTest1, lengthsTest1 = createStreams(testDataSet1)

#X1 = [[0.5], [1.0], [-1.0], [0.42], [0.24]]
#X2 = [[0.5], [1.0], [-1.0], [0.42], [0.24]]

#X = np.concatenate([X1, X2])
#lengths = [len(X1), len(X2)]

modelFor0 = GaussianHMM(n_components=2, n_iter=100).fit(streamTrain0, lengthsTrain0)
#modelFor1 = GaussianHMM(n_components=16, n_iter=100).fit(streamTrain1, lengthsTrain1)
# modelFor0 = GaussianHMM(n_components=2, n_iter=200).fit(trainingDataSet0[0])
predictTraining = np.zeros(shape=(trainingDataSet.shape[0],1))
results0 = np.zeros(shape=(trainingDataSet.shape[0],1))
# results1 = np.zeros(shape=(trainingDataSet.shape[0],1))
for i in range(0, trainingDataSet.shape[0]):
    predict0Results0 = modelFor0.decode(trainingDataSet[i].reshape(trainingDataSet[i].shape[0], 1),algorithm='viterbi')[0]
    #predict0Results1 = modelFor1.score(trainingDataSet[i].reshape(trainingDataSet[i].shape[0], 1))
    results0[i] = predict0Results0
    # if predict0Results0 > predict0Results1:
    #     results0[i] = 0
    # else:
    #     results0[i] = 1

    # print(predict0[63])

shapeTra = trainingDataSet.shape[0]
shapeTra0 = trainingDataSet0[0].shape[0]
shapeTra1 = trainingDataSet1[0].shape[0]


modelFor1 = GaussianHMM(n_components=2, n_iter=100).fit(streamTrain1, lengthsTrain1)
results1 = np.zeros(shape=(shapeTra,1))
for i in range(0, shapeTra):
    predict0Results1 = modelFor1.decode(trainingDataSet[i].reshape(trainingDataSet[i].shape[0], 1),algorithm='viterbi')[0]
    results1[i] = predict0Results1

results0_0 = np.zeros(shape=(shapeTra0, 1))
results1_0 = np.zeros(shape=(shapeTra0, 1))
# results1 = np.zeros(shape=(shapeTra,1))
for i in range(0, shapeTra0):
    predict0Results0 = \
    modelFor0.decode(trainingDataSet0[0][i].reshape(trainingDataSet0[0][i].shape[0], 1), algorithm='viterbi')[0]
    # predict0Results1 = modelFor1.score(trainingDataSet[i].reshape(trainingDataSet[i].shape[0], 1))
    results0_0[i] = predict0Results0
    predict1Results0 = modelFor1.decode(trainingDataSet0[0][i].reshape(trainingDataSet0[0][i].shape[0], 1), algorithm='viterbi')[0]
    results1_0[i] = predict1Results0

results0_1 = np.zeros(shape=(shapeTra1, 1))
results1_1 = np.zeros(shape=(shapeTra1, 1))
# results1 = np.zeros(shape=(shapeTra,1))
for i in range(0, shapeTra1):
    predict0Results1 = \
    modelFor0.decode(trainingDataSet1[0][i].reshape(trainingDataSet1[0][i].shape[0], 1), algorithm='viterbi')[0]
    # predict0Results1 = modelFor1.score(trainingDataSet[i].reshape(trainingDataSet[i].shape[0], 1))
    results0_1[i] = predict0Results1
    predict1Results1 = modelFor1.decode(trainingDataSet1[0][i].reshape(trainingDataSet1[0][i].shape[0], 1), algorithm='viterbi')[0]
    results1_1[i] = predict1Results1

X = results0-results1
print("")