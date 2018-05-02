# 以下程序给出了如何将MNIST数据转换为TFRecord格式
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np


# 定义函数转化变量类型。
def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))  # 整型


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))  # 字符串型


# 将数据转化为tf.train.Example格式。
def _make_example(pixels, label, image):
    image_raw = image.tostring()
    #  将样例转换为Example Protocol Buffer, 并将所有的信息写入这个数据结构
    example = tf.train.Example(features=tf.train.Features(feature={
        'pixels': _int64_feature(pixels),
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example


# 读取mnist训练数据。
mnist = input_data.read_data_sets("../../datasets/MNIST_data",dtype=tf.uint8, one_hot=False)
images = mnist.train.images
labels = mnist.train.labels
# print(images.shape)  # (55000, 784)
pixels = images.shape[1]  # 784
num_examples = mnist.train.num_examples  # 55000
# print(type(images[0]))  # <class 'numpy.ndarray'>


# 1. 将输入转化成TFRecord格式并保存。
# 训练数据
filename = "output.tfrecords"
with tf.python_io.TFRecordWriter(filename) as writer:
    for index in range(num_examples):
        example = _make_example(pixels, labels[index], images[index])

        # SerializeToString(): 序列化这个message和以字符串的方式返回。
        # 注意，这是二进行字节，不是一个文本； 我们只使用str类型作为一个方便的容器。
        writer.write(example.SerializeToString())
print("TFRecord训练数据已保存。")

# 测试数据
images_test = mnist.test.images
labels_test = mnist.test.labels
pixels_test = images_test.shape[1]
num_examples_test = mnist.test.num_examples

# 输出包含测试数据的TFRecord文件。
with tf.python_io.TFRecordWriter("output_test.tfrecords") as writer:
    for index in range(num_examples_test):
        example = _make_example(
            pixels_test, labels_test[index], images_test[index])
        writer.write(example.SerializeToString())
print("TFRecord测试数据已保存。")
