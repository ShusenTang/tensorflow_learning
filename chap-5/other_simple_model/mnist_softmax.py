"""
A very simple MNIST classifier.
from【http://cwiki.apachecn.org/pages/viewpage.action?pageId=10029423】
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def main(trainstep):
    # Import data
    mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.matmul(x, W) + b

    # Define loss and optimizer
    y_ = tf.placeholder(tf.float32, [None, 10])


    cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y))
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
      # 初始化
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(trainstep):
            if i % 1000 == 0:
                validate_acc = sess.run(accuracy, feed_dict={x: mnist.validation.images, y_: mnist.validation.labels})
                print("After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        # Test trained model
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        test_acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        test_accuracy = sess.run(test_acc, feed_dict={x: mnist.test.images,
                                      y_: mnist.test.labels})
        print(("After %d training step(s), test accuracy using average model is %g" % (trainstep, test_accuracy)))





if __name__ == '__main__':
    main(trainstep=5000)