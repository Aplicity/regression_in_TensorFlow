import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Linear Regression Problem

X_data = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,
                    2.167,7.042,10.791,5.313,7.997,5.654,9.27,3.1,
                    6.83, 4.668, 8.9, 7.91, 5.7, 8.7, 3.1, 2.1])
Y_data = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
                   2.827,3.465,1.65,2.904,2.42,2.94,1.3, 1.84, 2.273, 3.2,
                   2.831, 2.92, 3.24, 1.35, 1.03])

split = 17
train_X ,test_X = X_data[:split], X_data[split:]
train_Y, test_Y = Y_data[:split], Y_data[split:]

n_samples = X_data.shape[0]
training_samples = train_X.shape[0]
testing_samples = test_X.shape[0]

# Feed dict is slow. TF introduced a new 'Pytorch-y' way to load data- the Dataset API
BATCH_SIZE = 4
train_data = (np.expand_dims(train_X,-1), np.expand_dims(train_Y, -1))
test_data = (np.expand_dims(test_X, -1), np.expand_dims(test_Y, -1))

# Create two Datasets, training and testing
train_data = tf.data.Dataset.from_tensor_slices(train_data)
test_data = tf.data.Dataset.from_tensor_slices(test_data)

# Loop
train_data = train_data.repeat()

# create an iterator of the correct shape and type
iter = tf.data.Iterator.from_structure(train_data.output_types, train_data.output_shapes)
train_init_op = iter.make_initializer(train_data)
test_init_op = iter.make_initializer(test_data)

features, labels = iter.get_next()

rn_ini = np.random

W = tf.Variable(rn_ini.randn(), name = 'Weight', trainable = True , dtype = tf.float64)
b = tf.Variable(rn_ini.randn(), name = 'Bais', trainable = True, dtype = tf.float64)

# construct model: y_hat = X * W + b
pred = tf.add( tf.multiply(features, W), b)

# cost function
cost = tf.reduce_sum( tf.pow( pred - labels, 2))
cost = cost / training_samples

# set optimizer
lr = 0.001
optimizer = tf.train.GradientDescentOptimizer(lr)
optimizer = optimizer.minimize( cost )

training_epochs = 100
display_step = 10

# initializer variable
init = tf.global_variables_initializer()
cost_history = []

# run
with tf.Session() as sess:
    sess.run(init)
    sess.run(train_init_op)

    for epoch in range(training_epochs):
        local_cost = 0
        for _ in range(training_samples):
            # X, Y = sess.run([features, labels])
            _, curr_cost = sess.run([optimizer, cost])
            local_cost += curr_cost
        cost_history.append(local_cost)

        if (epoch + 1) % display_step == 0:
            curr_cost = sess.run(cost)
            print('Epoch:{}, Cost:{}'.format(epoch, curr_cost))
    # cost plot
    plt.plot(np.arange(training_epochs), np.array(cost_history), 'o')
    plt.xlabel('epoch')
    plt.ylabel('y')
    plt.title('cost plot')
    plt.show()

    # switch to Test data
    sess.run(test_init_op)
    plt.plot(test_X, test_Y, 'ro', label = 'test data')
    pred_y_test = []
    test_cost = 0
    for i in range(testing_samples):
        t, pred_y = sess.run([cost * training_samples / testing_samples, pred])
        test_cost += t
        pred_y_test.append(pred_y)

    print('Test Cost :', test_cost)
    plt.plot(test_X, np.array(pred_y_test), label = 'Fitted line')
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('pred_y_test')
    plt.show()

    print('W:', sess.run(W))
    print('b:', sess.run(b))

# Batching
# train_dataset = train_dataset.shuffle(buffer_size=train_samples)
# train_dataset = train_dataset.batch(BATCH_SIZE)
# test_dataset = test_dataset.batch(test_samples)

# test_cost, pred_y_test = sess.run([cost * train_samples/test_samples, pred])
# plt.plot(test_X, sess.run(pred), label='Fitted line')

