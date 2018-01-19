#basic setup for nearest neighbour
import tensorflow as tf
import numpy
import random
import matplotlib.pyplot as plt
import csv

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
#train_x=numpy.asarray([100,200,300,400,500])
train_y=numpy.asarray(prices)
#train_y=numpy.asarray([1000,2000,3000,4000,5000])
n=numpy.size(train_x)

X = tf.placeholder("float")
Y = tf.placeholder("float")

out=tf.add(X,tf.negative(Y))
pred=tf.argmin(abs(out),0)

init =  tf.global_variables_initializer()

with tf.Session() as sess :
    sess.run(init)
    ind = sess.run(pred,feed_dict={X:train_x,Y:[1929]})
    print (ind)
print (train_y[ind])
