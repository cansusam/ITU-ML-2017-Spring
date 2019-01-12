# Lecture:  BLG 527E Machine Learning
# Term:     2018 - Spring
# Student:  Omercan Susam
# ID:       504162517

import math
import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from os.path import dirname, abspath
import os
import time

cwd = os.getcwd()
timestr = time.strftime("%Y%m%d-%H%M%S")
d = dirname(dirname(abspath(__file__)))
# write results
the_file = open('hw3/results-'+ timestr +'-.txt', 'w');

classNumber = 10

with tf.name_scope('input'):
    testing_df = pd.read_csv(d + '/data/optdigits.tes',
                             header=None)
    X_testing, y_testing = testing_df.loc[:, 0:63], testing_df.loc[:, 64]

    training_df = pd.read_csv(d + '/data/optdigits.tra',
                              header=None)
    X_training, y_training = training_df.loc[:, 0:63], training_df.loc[:, 64]


def get_normed_mean_cov(X):
    X_std = StandardScaler().fit_transform(X)
    X_mean = np.mean(X_std, axis=0)
    X_cov = (X_std - X_mean).T.dot((X_std - X_mean)) / (X_std.shape[0] - 1)
    return X_std, X_mean, X_cov


X_train, _, _ = get_normed_mean_cov(X_training)
X_test, _, _ = get_normed_mean_cov(X_testing)

X_train = X_train.reshape(-1, 8, 8, 1)
X_test = X_test.reshape(-1, 8, 8, 1)

y_train = tf.keras.utils.to_categorical(y_training, 10)
y_test = tf.keras.utils.to_categorical(y_testing, 10)

optimizers =[
'sgd',
'rmsprop',
'adagrad',
'adadelta',
'adam',
'adamax',
'nadam'
]

for opt in optimizers:
    with tf.Session() as sess:
        # variables need to be initialized before we can use them
        sess.run(tf.global_variables_initializer())

        # create log writer object
        writer = tf.summary.FileWriter('hw3/logs')
        writer.add_graph(sess.graph)
        model = tf.keras.models.Sequential()

        with tf.name_scope('hidden') as scope:
            model.add(tf.keras.layers.Convolution2D(kernel_size=(3, 3), filters= 32, input_shape=(8,8,1),
                                activation='relu',
                                padding = 'valid'))
        with tf.name_scope('hidden2') as scope:
            model.add(tf.keras.layers.Convolution2D(kernel_size=(3, 3), filters= 32, padding='same'))
        with tf.name_scope('hidden3') as scope:
            model.add(tf.keras.layers.Convolution2D(kernel_size=(3, 3), filters= 32, padding='same'))
        with tf.name_scope('hidden2') as scope:
            model.add(tf.keras.layers.Convolution2D(kernel_size=(3, 3), filters= 32, padding='same'))
        with tf.name_scope('hidden3') as scope:
            model.add(tf.keras.layers.Convolution2D(kernel_size=(3, 3), filters= 32, padding='same'))
        with tf.name_scope('pool') as scope:
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2), data_format='channels_last'))
        with tf.name_scope('dropout') as scope:
            model.add(tf.keras.layers.Dropout(0.3))
        with tf.name_scope('flatten') as scope:
            model.add(tf.keras.layers.Flatten())
        with tf.name_scope('dense') as scope:
            model.add(tf.keras.layers.Dense(128, activation='relu'))
        with tf.name_scope('dropout') as scope:
            model.add(tf.keras.layers.Dropout(0.5))
        with tf.name_scope('flatten') as scope:
            model.add(tf.keras.layers.Flatten())
        with tf.name_scope('dense') as scope:
            model.add(tf.keras.layers.Dense(128, activation='relu'))
        with tf.name_scope('dense') as scope:
            model.add(tf.keras.layers.Dense(10, activation='softmax'))


        with tf.name_scope('train'):
            model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
            hist = model.fit(X_train,y_train, batch_size=128, epochs=25, validation_data=(X_test,y_test))
        scores = model.evaluate(X_test, y_test, verbose=0)
        p = model.predict(X_test, verbose=1)
        #print(sess.run(hist))
        tf.summary.scalar('histogram', p)
        merged = tf.summary.merge_all()
        writer.close()

    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Loss Rate', size=14)
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.legend(['Training', 'Testing'], loc='upper right')
    plt.savefig('hw3/Model_loss'+ opt + "-" + timestr)
    #plt.show()
    plt.clf()

    plt.plot(hist.history['acc'])
    plt.plot(hist.history['val_acc'])
    plt.title('Accuracy Rate', size=14)
    plt.ylabel('Accuracy %')
    plt.xlabel('Epochs')
    plt.legend(['Training','Testing'], loc='lower right')
    plt.savefig('hw3/Model_acc'+ opt +"-" + timestr)
    #plt.show()
    plt.clf()

    # Confusion matrix calculations
    results = np.transpose(np.argmax(p, axis=1))

    #c= tf.confusion_matrix(y_test,p)

    testLabels = np.array(y_testing)

    freq = np.unique(testLabels, return_counts=True)[1]

    matchingClassNumbers = np.zeros(classNumber)
    freqResults = np.zeros((classNumber))
    for x in range(0, classNumber):
        matchingClassIndex = (results == x) & (testLabels == x)
        matchingClassNumbers[x] = sum(matchingClassIndex)
        freqResults[x] = matchingClassNumbers[x] / freq[x]

    freqMatrix = np.zeros((classNumber, classNumber))
    the_file.write(opt + "\nClass\tAccuracy\t\n")
    for x in range(0, classNumber):
        subsetOfMust = results[testLabels == x]
        for y in range(0, classNumber):
            freqMatrix[x, y] = sum(subsetOfMust == y)
        print(x, "\t", float("%0.3f" % (freqResults[x] * 100)), "%\t")
        the_file.write("%d\t" % x)
        the_file.write("%0.3f%%\t\n" % (freqResults[x] * 100))

    # confusion matrix calculation
    confusionMatrix = np.zeros((classNumber, classNumber))
    for x in range(0, testing_df.shape[0]):
        prediction = int(results[x]);
        actual = int(testLabels[x]);
        confusionMatrix[actual, prediction] = confusionMatrix[actual, prediction] + 1 / freq[actual];

    # plot confusion matrix
    fig, ax1 = plt.subplots(1, 1)
    ax1.imshow(confusionMatrix, cmap='Blues', alpha=.9, interpolation='nearest')
    ax1.set_xticks(np.arange(0, 10, 1))
    ax1.set_yticks(np.arange(0, 10, 1))
    fig.suptitle("Confusion Matrix-")
    fig.savefig("hw3/Confusion Matrix-" + opt + "-" + timestr)
    plt.clf()


