import time
import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
import forward
import train


# 每隔 EVAL_INTERVAL_SECS 秒加载一次最新的模型，并在测试数据集上测试正确率
EVAL_INTERVAL_SECS = 10

def evaluate(mnist):
    # tf.Graph()生成新的计算图
    # as_default()返回一个上下文管理器,使得这个图成为当前默认的图，
    # 通过with关键字和这个方法,来让这个代码块内创建的操作(ops)添加到这个图里面
    with tf.Graph().as_default(): #  as g:
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE], name='y-input')
        validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}

        y = forward.inference(x, None)
        # tf.argmax(vector, 1)：返回的是vector中的最大值的索引号
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 使用重命名的方式来加载模型
        variable_averages = tf.train.ExponentialMovingAverage(train.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        # saver = tf.train.Saver() # 以上三行也可以用本行代替只是准确率可能会有细微下降

        while True:
            with tf.Session() as sess:
                # get_checkpoint_state会通过checkpoint文件自动找到目录中的最新的模型
                ckpt = tf.train.get_checkpoint_state(train.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict=validate_feed)
                    print("After %s training step(s), validation accuracy = %g" % (global_step, accuracy_score))
                else:
                    print('No checkpoint file found')
                    return
            time.sleep(EVAL_INTERVAL_SECS)


def main():
    mnist = train.input_data.read_data_sets("../datasets/MNIST_data", one_hot=True)
    evaluate(mnist)


if __name__ == '__main__':
    main()
