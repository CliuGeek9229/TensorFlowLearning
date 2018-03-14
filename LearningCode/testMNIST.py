import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

# number 1 to 10 data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


# add one more layer and return the output of this layer
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
        return outputs


def compute_accuracy(v_xs, v_ys):  # 提升准确度
    global prediction
    y_pre = sess.run(prediction, feed_dict={xs: v_xs})#这个地方的prediction预测出来的是一组概率值
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys,
                                                                 1))  # argmax返回沿着某个维度最大值的位置 1代表返回每一列的最大值的位置索引   0代表返回每一行的最大值的位置索引  tf.equal(A, B)是对比这两个矩阵或者向量的相等的元素，如果是相等的那就返回True，反正返回False，返回的值的矩阵维度和A是一样的
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # tf.cast类型转换
    result = sess.run(accuracy, feed_dict={xs: v_xs, ys: v_ys})
    return result  # 百分比


# define placeholder for inputs to network
xs = tf.placeholder(tf.float32, [None, 784])  # 不规定它有多少个simple  但是要规定点的个数  28*28=784
ys = tf.placeholder(tf.float32, [None, 10])  # 10个输出  如果是3的话，输出结果为[0,0,0,1,0,0,0,0,0,0]

# add output layer
prediction = add_layer(xs, 784, 10, activation_function=tf.nn.softmax)  # 注意这个地方的activation  softmax是用来做classfication的

# the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys * tf.log(prediction),
                                              reduction_indices=[1]))  # loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
# softmax 和  cross_entropy可以配合使用生成分类算法

sess = tf.Session()
# important  step
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)  # 从下载好的数据集中提取100个  也就是随机梯度下降的方法
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))#测试集   我现在想显示的就是在已经训练好的神经网络上a

