import tensorflow as tf
import numpy
import random
import matplotlib.pyplot as plt
import csv

alpha=0.0000001
cycles=100

prices=[]
sqft=[]
first=True
with open('RealEstate.csv','r') as f :
    reader=csv.reader(f)
    for row in reader :
        if first==True :
            first=False
        else :
            prices.append((int)(row[2][:-3]))
            sqft.append((int)(row[5]))

train_x=numpy.asarray(sqft)
train_y=numpy.asarray(prices)
n=numpy.size(train_x)

print(n)

W=tf.Variable(numpy.random.randn(),name="weight")
b=tf.Variable(numpy.random.randn(),name="bias")

X = tf.placeholder("float")
Y = tf.placeholder("float")

out=tf.add(tf.multiply(X,W),b)

error=tf.reduce_sum(tf.pow(out-Y,2))/(2*n)
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


