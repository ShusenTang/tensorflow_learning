"""
A very simple MNIST classifier.
from【http://cwiki.apachecn.org/pages/viewpage.action?pageId=10029431】
"""

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf

def conv2d(x, W):
    """conv2d returns a 2d convolution layer with full stride."""
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """max_pool_2x2 downsamples a feature map by 2X."""
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape):
    """weight_variable generates a weight variable of a given shape."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    """由于我们使用 ReLU神经元，因此较好的做法是用一个较小的正数来初始化偏置量，以避免神经元节点输出恒为0"""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def main(trainstep):
    # Import data
    mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)

    # Create the model
    x = tf.placeholder(tf.float32, [None, 784])
    y_ = tf.placeholder(tf.float32, [None, 10])


    # 第一层卷积
    # 它由一个卷积接一个最大池化构成。卷积在每5x5的patch中算出32个特征。它的权重张量形状为[5, 5, 1, 32]。
    # 前两个维度是patch大小，第三个维度是图片颜色输入通道的数量（灰度图片为1，彩色图片为3），最后一个是输出通道的数量。
    # 我们还将为每个输出通道添加一个偏置量。
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_image = tf.reshape(x, [-1, 28, 28, 1]) # 为了将层应用于图中，我们首先reshape x形成4d张量，第二和第三维对应于图片的宽度和高度，最后的尺寸对应于颜色通道的数量
    # 然后我们将x_image与权重张量进行卷积，添加偏置量，应用于ReLU函数，最后再加入最大池化。该max_pool_2x2方法将图像大小减小到14x14
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积
    # 为了构建一个深层次的网络，我们堆叠几层这种类型的模块。第二层将为每个5x5patch提取64特征。
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)


    # 全连接层
    # 现在图像尺寸已经缩小到7x7，我们添加了一个具有1024个神经元的全连接层，以便对整个图像进行处理。我们从池化层将张量reshape成一些向量，然后乘以权重矩阵，添加偏置量并应用于ReLU
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)


    #Dropout
    # 为了避免过拟合，我们在输出层之前加入dropout 。我们创建一个占位符表示在神经元dropuot时保持不变的概率。
    # 这样可以让我们在训练过程中应用dropout，并在测试过程中将其关闭。TensorFlow的tf.nn.dropoutop除了
    # 可以屏蔽神经元的输出外，还可以自动处理神经元输出scale，所以用dropout的时候可以不用考虑任何额外scale
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # 输出层
    # 最后，我们添加一层，就像上面提到的softmax回归层一样。
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    with tf.Session() as sess:
      # 初始化
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(trainstep):
            batch = mnist.train.next_batch(50)
            if i % 500 == 0:
                validate_accuracy = sess.run(accuracy,feed_dict={
                    x: mnist.validation.images, y_: mnist.validation.labels, keep_prob: 1.0})
                print("step %d, validation accuracy is %g" % (i, validate_accuracy))
            train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}) # 使用了dropout

        print("test accuracy is %g" % accuracy.eval(feed_dict={
            x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


if __name__ == '__main__':
    main(trainstep=3000)