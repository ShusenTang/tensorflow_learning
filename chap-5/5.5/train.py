import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import forward
import os

BATCH_SIZE = 100
LEARNING_RATE_BASE = 0.8
LEARNING_RATE_DECAY = 0.99
REGULARIZATION_RATE = 0.0001
TRAINING_STEPS = 30000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH="model_saved/"
MODEL_NAME="model.ckpt"


def train(mnist):

    x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE], name='x-input')
    y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE], name='y-input')

    regularizer = tf.contrib.layers.l2_regularizer(REGULARIZATION_RATE)
    y = forward.inference(x, regularizer) # 使用forward中定义的前向传播函数
    global_step = tf.Variable(0, trainable=False)

    # 滑动平均
    variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    variables_averages_op = variable_averages.apply(tf.trainable_variables())

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1)) # argmax得到正确答案(即1)的编号
    cross_entropy_mean = tf.reduce_mean(cross_entropy)
    loss = cross_entropy_mean + tf.add_n(tf.get_collection('losses')) # forward中将正则化损失加入了losses集合

    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        mnist.train.num_examples / BATCH_SIZE, LEARNING_RATE_DECAY,
        staircase=True)
    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
    # 同时进行反向传播更新参数和更新每一个参数的滑动平均值。一次完成多个操作，可以使用control_dependencies和group两种机制
    # 以下两行等价于 train_op = tf.group(train_step, variables_averages_op)
    with tf.control_dependencies([train_step, variables_averages_op]):
        train_op = tf.no_op(name='train')

    # 初始化TensorFlow持久化类
    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        for i in range(TRAINING_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)
            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if (i+1) % 1000 == 0:   # 每1000轮保存一次模型
                # 这里只输出当前模型在训练集上的loss，在验证集上的loss会有一个单独的程序来生成
                print("After %d training step(s), loss on training batch is %g." % (step, loss_value))
                # 这里给了global_step参数，可以使文件名末尾加上训练轮数如 mnist_model.ckpt-1000
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)


def main(argv = None):
    mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)
    train(mnist)


if __name__ == '__main__':
    tf.app.run()

