import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from sklearn.datasets import load_digits
#from sklearn.cross_validation import train_test_split  新版本中的使用方法为下一行所示
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load data
digits = load_digits()
X = digits.data#数据集
y = digits.target#类别标签
y = LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=.3)#70%做训练30%做测试
#print(len(X_train),len(X_test),len(y_train),len(y_test))
#add one more layer and return the output of this layer
def add_layer(inputs,in_size,out_size,n_layer,activation_function = None):
    #layer_name = 'layer%s' % n_layer
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases
    Wx_plus_b = tf.nn.dropout(Wx_plus_b,keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    #tf.summary.histogram(layer_name+'/outputs',outputs)
    return outputs
#define placeholder for inputs to network
keep_prob = tf.placeholder(tf.float32)
xs = tf.placeholder(tf.float32,[None,64])#8X8
ys = tf.placeholder(tf.float32,[None,10])

#add output layer
l1 = add_layer(xs,64,50,'l1',activation_function=tf.nn.tanh)
prediction = add_layer(l1,50,10,'l2',activation_function=tf.nn.softmax)

#the loss between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),reduction_indices=[1]))
tf.summary.scalar('loss',cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.6).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()
#summary writer goes in here
train_writer = tf.summary.FileWriter("logs/train",sess.graph)
test_writer = tf.summary.FileWriter("logs/test",sess.graph)


sess.run(tf.global_variables_initializer())

for i in range(500):
    sess.run(train_step,feed_dict={xs:X_train,ys:y_train,keep_prob:0.5})#保持百分比   如果想drop40%，那么这个值为0.6
    if i % 50 ==0:
        #record loss
        train_result = sess.run(merged,feed_dict={xs:X_train,ys:y_train,keep_prob:1})#记录result时不想drop任何东西
        test_result = sess.run(merged,feed_dict={xs:X_test,ys:y_test,keep_prob:1})
        train_writer.add_summary(train_result,i)
        test_writer.add_summary(test_result,i)