from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np
from os.path import dirname, abspath
import os
import time

cwd = os.getcwd()
timestr = time.strftime("%Y%m%d-%H%M%S")

d = dirname(dirname(abspath(__file__)))


classNumber = 10
featureNumber = 64

trainingDataSet = np.loadtxt(d+'/data/optdigits.tra', delimiter=',');
testDataSet = np.loadtxt(d+'/data/optdigits.tes', delimiter=',');
#trainingDataSet = tf.convert_to_tensor(trainingDataSet_, np.int32)
#testDataSet = tf.convert_to_tensor(testDataSet_, np.int32)


# write results
# the_file = open('hw3/results-'+ timestr +'-.txt', 'w');

# store labels of each sample
trainingLabels = trainingDataSet[:, 64]
testLabels = testDataSet[:, 64]

#  remove lables from features
trainingDataSet = trainingDataSet[:, :64]
testDataSet = testDataSet[:, :64]

trainingLabels = np.reshape(trainingLabels, (-1, 1))
testLabels = np.reshape(testLabels, (-1, 1))

EPOCHS = trainingDataSet.shape[0]
x, y = tf.placeholder(tf.float32, shape=[None,64]), tf.placeholder(tf.float32, shape=[None,1])
dataset = tf.data.Dataset.from_tensor_slices((x, y))



iter = dataset.make_initializable_iterator()
features, labels = iter.get_next()
with tf.Session() as sess:
#     initialise iterator with train data
    sess.run(iter.initializer, feed_dict={ x: trainingDataSet, y: trainingLabels})
    for _ in range(EPOCHS):
        sess.run([features, labels])
#     switch to test data
    sess.run(iter.initializer, feed_dict={ x: testDataSet, y: testLabels})
    print(sess.run([features, labels]))