# Lecture:  BLG 527E Machine Learning
# Term:     2018 - Spring
# Student:  Omercan Susam
# ID:       504162517

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from os.path import dirname, abspath
import os
import time

cwd = os.getcwd()
timestr = time.strftime("%Y%m%d-%H%M%S")
d = dirname(dirname(abspath(__file__)))
# write results
the_file = open('hw3/results-hw3-q1-'+ timestr +'-.txt', 'w');


classifiers = [
    DecisionTreeClassifier(),
DecisionTreeClassifier(max_depth=3),
DecisionTreeClassifier(criterion="entropy"),
DecisionTreeClassifier(max_leaf_nodes=3),
DecisionTreeClassifier(max_depth=3, max_leaf_nodes=3, criterion="entropy"),
MLPClassifier(alpha=1),
MLPClassifier(activation="logistic",alpha=1),
MLPClassifier(solver='sgd',alpha=1),
MLPClassifier(learning_rate="adaptive" ,alpha=1),
MLPClassifier(activation="logistic",solver='sgd',learning_rate="adaptive" ,alpha=1)
]

classNumber = 10
featureNumber = 64

trainingDataSet = np.loadtxt(d+'/data/optdigits.tra', delimiter=',');
testDataSet = np.loadtxt(d+'/data/optdigits.tes', delimiter=',');

# store labels of each sample
trainingLabels = trainingDataSet[:, 64]
testLabels = testDataSet[:, 64]

#  remove lables from features
trainingDataSet = trainingDataSet[:, :64]
testDataSet = testDataSet[:, :64]

counter = 0
for clf in classifiers:
    c = clf.fit(trainingDataSet, trainingLabels)
    name = clf.__class__.__name__
    # train_predictions = clf.predict_proba(df_test)
    results = clf.predict(testDataSet)
    # sub.to_csv(name + '.csv', float_format='%.8f', index=False)
    # find results for accuracy
    print(name, "\n", "Class\t", "Accuracy\t")
    the_file.write(name + "\nClass\tAccuracy\t\n")

    # class frequencies
    freq = np.unique(testLabels, return_counts=True)[1]

    matchingClassNumbers = np.zeros(classNumber)
    freqResults = np.zeros((classNumber))
    for x in range(0, classNumber):
        matchingClassIndex = (results == x) & (testLabels == x)
        matchingClassNumbers[x] = sum(matchingClassIndex)
        freqResults[x] = matchingClassNumbers[x] / freq[x]

    freqMatrix = np.zeros((classNumber, classNumber))
    for x in range(0, classNumber):
        subsetOfMust = results[testLabels == x]
        for y in range(0, classNumber):
            freqMatrix[x, y] = sum(subsetOfMust == y)
        print(x, "\t", float("%0.3f" % (freqResults[x] * 100)), "%\t")
        the_file.write("%d\t" % x)
        the_file.write("%0.3f%%\t\n" % (freqResults[x] * 100))

    # confusion matrix calculation
    confusionMatrix = np.zeros((classNumber, classNumber));
    for x in range(0, testDataSet.shape[0]):
        prediction = int(results[x]);
        actual = int(testLabels[x]);
        confusionMatrix[actual, prediction] = confusionMatrix[actual, prediction] + 1 / freq[actual];

    # plot confusion matrix
    fig, ax1 = plt.subplots(1, 1)
    ax1.imshow(confusionMatrix, cmap='Blues', alpha=.9, interpolation='nearest')
    ax1.set_xticks(np.arange(0, 10, 1))
    ax1.set_yticks(np.arange(0, 10, 1))
    fig.suptitle("Confusion Matrix-" + name)
    fig.savefig("hw3/Confusion Matrix-hw3-q1" + str(counter) + "-" + name + "-" + timestr)
    counter = counter + 1