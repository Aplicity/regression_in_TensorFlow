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
train_X, test_X = X_data[:split], X_data[split:]
train_Y, test_Y = Y_data[:split], Y_data[split:]

n_samples = X_data.shape[0]
training_samples = train_X.shape[0]
testing_samples = test_X.shape[0]

# Define graph inputs

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

rn_ini = np.random

# Define model parameter variable
W = tf.Variable(rn_ini.randn(), name = 'Weight', trainable = True)
b = tf.Variable(rn_ini.randn(), name = 'Bais', trainable = True)

# y_hat = X * W + b
pred = tf.add( tf.multiply(X, W), b)

# cost function
cost = tf.reduce_sum( tf.pow(pred - Y, 2))
cost = cost / training_samples

# set optimizer
lr = 0.001
training_epochs = 500
display_step = 100

optimizer = tf.train.GradientDescentOptimizer( learning_rate = lr)
optimizer = optimizer.minimize(cost)

# initializer variable
init = tf.global_variables_initializer()
cost_history = []

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(training_epochs):
        local_cost = 0
        for (x,y) in zip(train_X, train_Y):
            _, cost_curr = sess.run([optimizer, cost], feed_dict = {X:x , Y:y})
            local_cost += cost_curr
        cost_history.append(local_cost / training_samples)

        if (epoch +1 ) % 100 == 0:
            cost_curr = sess.run( cost, feed_dict = {X:x, Y:y})
            print('Epoch:{} , Cost :{:4}'.format(epoch, cost_curr))

    # training plot
    plt.plot(train_X, train_Y, 'ro', label = 'Training data')
    plt.plot(train_X, sess.run(pred, feed_dict = {X : train_X}), label = 'fitted line')
    plt.legend()
    plt.title('training plot')
    plt.show()

    # testing plot
    plt.plot(test_X, test_Y, 'ro', label = 'Testing data')
    plt.plot(test_X, sess.run(pred, feed_dict = {X: test_X}), label = 'fitted line')
    plt.legend()
    plt.title('testing plot')
    plt.show()

    # cost plot
    plt.scatter(np.arange(training_epochs), np.array(cost_history))
    plt.xlabel('epoch')
    plt.ylabel('cost')
    plt.title('cost plot')
    plt.show()
