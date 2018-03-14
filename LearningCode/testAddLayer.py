import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt #可视化
#定义一个神经层
#add one more layer and return the output of this layer
def add_layer(inputs,in_size,out_size,activation_function = None):
        #默认的none为线性方程
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))#矩阵  权重。大写开头意思为矩阵 random_normal表示为正态分布的随机数
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)#首字母小写，列表
    Wx_plus_b = tf.matmul(inputs,Weights) + biases #matmul相乘加biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs
#make up some real data
x_data = np.linspace(-1,1,300)[:,np.newaxis]#再列上加了一个维度，现在是300行1列的二维数组
noise = np.random.normal(0,0.05,x_data.shape)#噪点  使得数据更加像一个真实的数据
y_data = np.square(x_data) - 0.5 + noise#
#define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None,1])#[None,1]为格式shape   即不限定行  列为1
#none表示无论给多少个例子都是可以的
ys = tf.placeholder(tf.float32,[None,1])

#add hidden layer
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)#隐藏层  输入  输入size 输出size 激励函数
#add outpt layer
predition = add_layer(l1,10,1,activation_function=None)#输出层
#the error between prediction and real data
loss = tf.reduce_mean(tf.reduce_sum(tf.square(ys - predition),
                     reduction_indices=[1]))
#对每一个误差平方求和然后再求平均值 reduction_indices = [1]按行求和  reduction_indices = [0] 按列求和
train_step = tf.train.GradientDescentOptimizer(0.09).minimize(loss)#梯度下降  学习效率0.1 要做的是minimize这个loss
#important step
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()#不会暂停show  不断刷新红线
plt.show()

for i in range(1000):
    #training
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50:
        #to see the step improvement
        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = sess.run(predition,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=3)
        plt.pause(0.1)
