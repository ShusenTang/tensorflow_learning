import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
# 因为需要在变量定义的时候加上日志输出，所以这里不用之前的mnist_inference.py




''' 1. 生成变量监控信息并定义生成监控信息日志的操作 '''
SUMMARY_DIR = "log/simple_example-11.3"
BATCH_SIZE = 100
TRAIN_STEPS = 3000


# 生成变量监控信息并定义生成监控信息日志的操作。
# 其中var给出了需要记录的张量，name给出了在可视化结果中显示图表的名称，这个名称一般与变量名一致。
def variable_summaries(var, name):
    # 将生成监控信息的操作放到同一个命名空间下
    with tf.name_scope('summaries'):
        # tf.summary.histogram记录张量元素的取值分布。将summary写入日志文件后，在tensorboard网页的HISTOGRAM栏
        # 和DISTRIBUTION栏下都会出现对应名称的图表。
        # 与其他操作类似，需要sess.run后才能真正生成
        tf.summary.histogram(name, var)

        # 计算变量的平均值，并定义生成平均值信息日志的操作。
        # 记录变量平均值信息的日志标签名为'mean/' + name，其中mean是命名空间，相同命名空间中的监控指标会被整合到同一栏中。
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean/' + name, mean)

        # 标准差
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev/' + name, stddev)




'''2. 生成一层全链接的神经网络'''
def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    # 将同一层神经网络放在统一的命名空间下
    with tf.name_scope(layer_name):
        # 声明权重和偏置，并调用生成权重个偏置监控信息日志的函数
        with tf.name_scope('weights'):
            weights = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
            variable_summaries(weights, layer_name + '/weights')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.constant(0.0, shape=[output_dim]))
            variable_summaries(biases, layer_name + '/biases')

        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            # 记录神经网络输出节点在经过激活函数之前的分布
            tf.summary.histogram(layer_name + '/pre_activations', preactivate)
        activations = act(preactivate, name='activation')

        # 记录神经网络节点输出在经过激活函数之后的分布
        # 注意观察tensorboard：layer1使用了relu激活函数后，所有小于0的值都会被设置成0
        tf.summary.histogram(layer_name + '/activations', activations)
        return activations


'''3. 主函数'''
def main(unused_arg):
    mnist = input_data.read_data_sets("../chap-5/datasets/MNIST_data", one_hot=True)

    with tf.name_scope('input'):
        x = tf.placeholder(tf.float32, [None, 784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')

    # 将输入向量还原成图片的像素矩阵，并通过tf.summary.image函数将当前图片信息写入日志
    with tf.name_scope('input_reshape'):
        image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
        tf.summary.image('input', image_shaped_input, 10)

    hidden1 = nn_layer(x, 784, 500, 'layer1') # layer1使用了默认的relu
    y = nn_layer(hidden1, 500, 10, 'layer2', act=tf.identity) # layer2没使用激活函数

    # 定义交叉熵并定义生成交叉熵监控日志的操作
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=y, labels=y_))
        tf.summary.scalar('cross_entropy', cross_entropy)

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    # 定义准确率并定义生成准确率监控日志的操作
    # 如果sess.run时给定的数据是训练batch，那么得到的准确率就是在这个batch上的准确率；如果给定的是验证集或者测试集，
    # 那么得到的准确率就是当前模型在验证或测试时的准确率
    with tf.name_scope('accuracy'):
        with tf.name_scope('correct_prediction'):
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        with tf.name_scope('accuracy'):
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)

    # 与TensorFlow其他操作一样，tf.summary.scalar、histogram、image等写日志相关函数需要通过sess.run来调用这些函数，但是一一调用很麻烦，
    # 于是用tf.summary.merge_all来整理所有日志操作，这样sess.run中只需调用一次即可将日志写入文件。
    merged = tf.summary.merge_all()

    with tf.Session() as sess:
        summary_writer = tf.summary.FileWriter(SUMMARY_DIR, sess.graph) # 初始化日志writer并写入计算图
        tf.global_variables_initializer().run()

        for i in range(TRAIN_STEPS):
            xs, ys = mnist.train.next_batch(BATCH_SIZE)

            # 运行训练步骤以及所有的日志生成操作，得到这次运行的日志。
            summary, _ = sess.run([merged, train_step], feed_dict={x: xs, y_: ys})
            # 将得到的所有日志写入日志文件，这样TensorBoard程序就可以拿到这次运行所对应的运行信息。
            summary_writer.add_summary(summary, i)

    summary_writer.close()

if __name__ == '__main__':
    tf.app.run()