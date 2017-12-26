import tensorflow as tf
import numpy
import random
import matplotlib.pyplot as plt

alpha=0.01
cycles=100

train_x=numpy.asarray([1,2,3,4,5,6,7,8,9,10])
train_y=numpy.asarray([1,2,3,3,2,6,7,4,9,10])
n=numpy.size(train_x)

W=tf.Variable(numpy.random.randn(),name="weight")
b=tf.Variable(numpy.random.randn(),name="bias")

X = tf.placeholder("float")
Y = tf.placeholder("float")

out=tf.add(tf.multiply(X,W),b)

error=tf.reduce_sum(tf.pow(out-Y,2))
gd=tf.train.GradientDescentOptimizer(alpha).minimize(error)

init=tf.global_variables_initializer()

sess=tf.Session()
sess.run(init)

print (sess.run(W),(sess.run(b)))

for c in range(cycles) :
    for (x,y) in zip(train_x,train_y) :
        sess.run(gd,feed_dict={X:x,Y:y})
    print (sess.run(W),(sess.run(b)))

plt.plot(train_x, train_y, 'ro', label='Data')
plt.plot(train_x, sess.run(W) * train_x + sess.run(b), label='Model')
plt.legend()
plt.show()
