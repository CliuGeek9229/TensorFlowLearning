import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np


#add one more layer and return the output of this layer
def add_layer(inputs,in_size,out_size,n_layer,activation_function = None):
    layer_name = 'layer%s' % n_layer
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name = 'W')
            tf.summary.histogram(layer_name+'/Weights',Weights)
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)
            tf.summary.histogram(layer_name+'/biases',biases)
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.matmul(inputs,Weights) + biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)
            tf.summary.histogram(layer_name+'/outputs',outputs)
        return outputs

#make up some real data
x_data = np.linspace(-1,1,300)[:,np.newaxis]#再列上加了一个维度，现在是300行1列的二维数组
noise = np.random.normal(0,0.05,x_data.shape)#噪点  使得数据更加像一个真实的数据
y_data = np.square(x_data) - 0.5 + noise#

#define placeholder for inputs to network
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')

#add hidden layer
l1 = add_layer(xs,1,10,n_layer=1,activation_function=tf.nn.relu)
#add outpt layer
predition = add_layer(l1,10,1,n_layer=2,activation_function=None)

#the error between prediction and real data
with tf.name_scope('loss'):
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),
                     reduction_indices=[1]))
    tf.summary.scalar('loss',loss)#观看loss   loss代表着神经网络是否学到了东西  它是一个纯量的变化  不同于前面的那些东西
with tf.name_scope('train'):
    train_step = tf.train.GradientDescentOptimizer(0.09).minimize(loss)

sess = tf.Session()
merged = tf.summary.merge_all()

#新版本将tf.train.SummaryWriter()改为tf.summary.FileWriter()

writer = tf.summary.FileWriter("log1/",sess.graph)
#important step
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data})
        writer.add_summary(result,i)


#先将目录调整到pro1
#执行命令  tensorboard --logdir=log1
#D:\tensorflowLearning\pro1>tensorboard --logdir=log1
