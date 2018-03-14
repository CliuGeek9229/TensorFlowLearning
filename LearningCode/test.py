import tensorflow as tf
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(tf.__path__)
    print(tf.__version__)