
batchSize = 50
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
x = tf.placeholder(tf.float32, [None, 64],name="x-in")


def getActivations(layer,stimuli):
    units = sess.run(layer,feed_dict={x:np.reshape(stimuli,[1,64],order='F')})
    plotNNFilter(units)

def plotNNFilter(units):
    filters = units.shape[3]
    plt.figure(1, figsize=(20,20))
    n_columns = 6
    n_rows = math.ceil(filters / n_columns) + 1
    for i in range(filters):
        plt.subplot(n_rows, n_columns, i+1)
        plt.title('Filter ' + str(i))
        plt.imshow(units[0,:,:,i], interpolation="nearest", cmap="gray")


imageToUse = X_train[0]
plt.imshow(np.reshape(imageToUse,[8,8]), interpolation="nearest", cmap="gray")
plt.show()
getActivations(hidden_1,imageToUse)