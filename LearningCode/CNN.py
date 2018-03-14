import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#number 1 to 10
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)

def compute_accuracy(v_xs,v_ys):
    global prediction
    y_pre = sess.run(prediction,feed_dict={xs:v_xs,keep_prob:1})
    correct_prediction = tf.equal(tf.argmax(y_pre,1),tf.argmax(v_ys,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    result = sess.run(accuracy,feed_dict={xs:v_xs,ys:v_ys,keep_prob:1})
    return result

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)#tf.truncated_normal(shape, mean, stddev) :shape表示生成张量的维度，mean是均值，stddev是标准差  和一般的正太分布的产生随机数据比起来，这个函数产生的随机数与均值的差距不会超过两倍的标准差，但是一般的别的函数是可能的。
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1,shape=shape)#biase的初始为0.1的常量   形状是shape
    return tf.Variable(initial)

def conv2d(x,W):#x为输入的值，W为权重  也就是上边生成的weight
    #stride[1,x_movement,y_movement,1]
    #Must have strides[0] = strides[3] = 1
    #padding意为填充  same意思为不缩减   valid为缩减
    return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')  #二维的CNN

def max_pool_2X2(x):
    # stride[1,x_movement,y_movement,1]
    # Must have strides[0] = strides[3] = 1
    #为了防止跨度太大  丢失信息  所以做卷积的时候将stride的2变为1  再做pooling操作  此时步长设为2  这样就可以起到把图像压缩的效果了
    return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')

#define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,784])/255. #28X28
ys = tf.placeholder(tf.float32,[None,10])
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(xs,[-1,28,28,1])
#-1代表的含义是不用我们自己指定这一维的大小，函数会自动计算，但列表中只能存在一个-1。
#1的意思就是黑白的图片  只有一个channel
#print(x_image.shape) #[n_samples,28,28,1]

#conv1 layer
W_conv1 = weight_variable([5,5,1,32]) # patch 5x5 卷积核大小,insize = 1即image的厚度为1 outsize = 32image的高度为32
b_conv1 = bias_variable([32])
#hidden convolutional neural network layer 1
#同时搭载了一个非线性函数的处理   relu
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)#output size 28x28x32  解释：原图像以5x5的大小抽出做卷积，卷积高度从原来的1变成了32，原来是28x28的图像  以same的方式做卷积  所以做完之后仍是28x28的大小  因而变成了28x28x32
h_pool1 = max_pool_2X2(h_conv1)                      #output size 14x14x32  解释：pooling在convolutional的基础上再做操作，这时候我们将步长设置为2以达到压缩图像长宽的目的，但是并未对高度做压缩设置，所以输出为14x14x32

#conv2 layer
W_conv2 = weight_variable([5,5,32,64]) # patch 5x5,insize = 32 outsize = 64image的高度为32
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)#output size 14x14x64
h_pool2 = max_pool_2X2(h_conv2)                      #output size 7x7x64

#fonc1 layer
#fully connnected
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])
#将pooling2的结果变平
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64]) #[n_samples,7,7,64]->>[n_samples,7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

#fonc2 layer
#output layer
W_fc2 = weight_variable([1024,10])
b_fc2 = bias_variable([10])
#在输出层我们用softmax的函数做classfication的处理
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

#the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)#这里改用AdamOptimizer  因为对于这么庞大的系统，使用AdamOptimizer的优化器比较好
#AdamOptimizer的learning rate使用更小的一个值   1e-4  也就是0.0001
sess = tf.Session()

#important step
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)
    sess.run(train_step,feed_dict={xs:batch_xs,ys:batch_ys,keep_prob:0.5})
    if i%50 == 0:
        print(compute_accuracy(mnist.test.images[:1000],mnist.test.labels[:1000]))

