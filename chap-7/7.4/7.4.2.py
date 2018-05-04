import tensorflow as tf
import numpy as np

# 数据集参数
# image_size = 299          # 定义神经网络输入层图片的大小。
batch_size = 100          # 定义组合数据batch的大小。
shuffle_buffer = 1000     # 定义随机打乱数据时buffer的大小。此数越大就越随机,但越占内存

# 网络参数
INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500
REGULARAZTION_RATE = 0.0001
# TRAINING_STEPS = 5000

# 输入数据使用7.1节生成的训练和测试数据。
# train.match_filenames_once函数和7.4.1中的placeholder机制类似, 也需要初始化,即用make_initializable_iterator()迭代器
train_files = tf.train.match_filenames_once("../7.1/output.tfrecords")
test_files = tf.train.match_filenames_once("../7.1/output_test.tfrecords")


# 解析一个TFRecord的解析方法。
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image_raw':tf.FixedLenFeature([],tf.string),
            'pixels':tf.FixedLenFeature([],tf.int64),
            'label':tf.FixedLenFeature([],tf.int64)
        })
    decoded_images = tf.decode_raw(features['image_raw'], tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    images = tf.reshape(retyped_images, [784])
    labels = tf.cast(features['label'], tf.int32)
    # pixels = tf.cast(features['pixels'],tf.int32)
    return images, labels


# 图像数据预处理程序(7.2.2程序)
def preprocess_for_train(image, height, width):
    if image.dtype != tf.float32:
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    # 将图片随机调整为神经网络输入层的大小。
    image = tf.image.resize_images(image, [height, width], method=np.random.randint(4))
    # 随机左右上下翻转
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)

    # image = tf.image.per_image_standardization(image) # 均值为0再clip后应该会丢失一些信息?
    image = tf.clip_by_value(image, 0.0, 1.0)
    return image


# 定义读取训练数据的数据集。
dataset = tf.data.TFRecordDataset(train_files)
dataset = dataset.map(parser)

# 上一次map得到了images和labels两个结果, 所以这个map有两个输入image和label
# dataset = dataset.map(
#     # lambda input: func(input) 返回func(input)
#     lambda image, label: (preprocess_for_train(image, image_size, image_size), label)
#     )

# 对数据进行shuffle和batching操作。
dataset = dataset.shuffle(shuffle_buffer).batch(batch_size)

# 数据集按以上方法重复处理重复NUM_EPOCHS次, 也间接指定了训练NUM_EPOCHS轮.
NUM_EPOCHS = 10
dataset = dataset.repeat(NUM_EPOCHS)

# 定义数据集迭代器。
# train.match_filenames_once函数和7.4.1中的placeholder机制类似, 也需要初始化,即用make_initializable_iterator()迭代器
iterator = dataset.make_initializable_iterator()
image_batch, label_batch = iterator.get_next()


# 定义神经网络的结构以及优化过程。
def inference(input_tensor, weights1, biases1, weights2, biases2):
    layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
    return tf.matmul(layer1, weights2) + biases2


weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))

weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

y = inference(image_batch, weights1, biases1, weights2, biases2)

# 计算交叉熵及其平均值
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=label_batch)
cross_entropy_mean = tf.reduce_mean(cross_entropy)

# 损失函数的计算
regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
regularization = regularizer(weights1) + regularizer(weights2)
loss = cross_entropy_mean + regularization

# 优化损失函数
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

# 定义测试用的Dataset。
test_dataset = tf.data.TFRecordDataset(test_files)
test_dataset = test_dataset.map(parser)
test_dataset = test_dataset.batch(batch_size)

# 定义测试数据上的迭代器。
test_iterator = test_dataset.make_initializable_iterator()
test_image_batch, test_label_batch = test_iterator.get_next()

# 定义测试数据上的预测结果。
test_logit = inference(test_image_batch, weights1, biases1, weights2, biases2)
predictions = tf.argmax(test_logit, axis=-1, output_type=tf.int32)

# 声明会话并运行神经网络的优化过程。
with tf.Session() as sess:
    # 初始化变量。
    sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

    # 初始化训练数据的迭代器。
    sess.run(iterator.initializer)

    # 循环进行训练，直到数据集完成输入、抛出OutOfRangeError错误。
    while True:
        try:
            sess.run(train_step)
        except tf.errors.OutOfRangeError:
            break

    test_results = []
    test_labels = []
    # 初始化测试数据的迭代器。
    sess.run(test_iterator.initializer)
    # 获取预测结果。
    while True:
        try:
            pred, label = sess.run([predictions, test_label_batch])
            test_results.extend(pred)
            test_labels.extend(label)
        except tf.errors.OutOfRangeError:
            break

# 计算准确率
correct = [float(y == y_) for (y, y_) in zip(test_results, test_labels)]
accuracy = sum(correct) / len(correct)
print("Test accuracy is:", accuracy)  # 0.9185