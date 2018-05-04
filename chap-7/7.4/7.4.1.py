import tensorflow as tf


# 1. 从数组创建数据集
def create_dataset_from_array(input_data = [1, 2, 3, 5, 8]):
    dataset = tf.data.Dataset.from_tensor_slices(input_data)

    # 定义迭代器。
    # 由于上面的数据集定义没有使用placeholder, 所以就用简单的make_one_shot_iterator(),
    # 否则就需要初始化, 即应该使用make_initializable_iterator()
    iterator = dataset.make_one_shot_iterator()

    # get_next() 返回代表一个输入数据的张量。
    x = iterator.get_next()
    y = x * x

    with tf.Session() as sess:
        for i in range(len(input_data)):
            print(sess.run(y))


# 2. 读取文本文件里的数据
def create_dataset_from_txt(input_files):
    # 从文本文件创建数据集。这里可以提供多个文件。
    # input_files = ["./test1.txt", "./test2.txt"]

    # TextLineDataset按每行一条数据读取
    dataset = tf.data.TextLineDataset(input_files)

    # 定义迭代器。
    iterator = dataset.make_one_shot_iterator()

    # 这里get_next()返回一个字符串类型的张量，代表文件中的一行。
    x = iterator.get_next()
    with tf.Session() as sess:
        for i in range(4):
            print(sess.run(x))

# 解析一个TFRecord的方法, 以7.1节的为例
def parser(record):
    features = tf.parse_single_example(
        record,
        features={
            'image_raw':tf.FixedLenFeature([],tf.string),
            'pixels':tf.FixedLenFeature([],tf.int64),
            'label':tf.FixedLenFeature([],tf.int64)
        })
    decoded_images = tf.decode_raw(features['image_raw'],tf.uint8)
    retyped_images = tf.cast(decoded_images, tf.float32)
    images = tf.reshape(retyped_images, [784])
    labels = tf.cast(features['label'],tf.int32)
    #pixels = tf.cast(features['pixels'],tf.int32)
    return images, labels

#  3. 解析TFRecord文件里的数据
def create_dataset_from_TFRecord(input_files):
    # 从TFRecord文件创建数据集。这里可以提供多个文件。
    # input_files = ["output.tfrecords"]
    dataset = tf.data.TFRecordDataset(input_files)

    # map(parser)函数表示对数据集中的每一条数据调用parser方法。
    dataset = dataset.map(parser)

    # 定义遍历数据集的迭代器。
    iterator = dataset.make_one_shot_iterator()

    # 读取数据，可用于进一步计算
    image, label = iterator.get_next()

    with tf.Session() as sess:
        for i in range(10):
            x, y = sess.run([image, label])
            print(y)


# 使用initializable_iterator来动态初始化数据集
# 前三种都使用了one_shot_iterator,此时数据集所有参数都已经确定,如果需要用palaceholder来初始化数据集,就要用到initializable_iterator
# 使用placeholder + feed_dict的方式可以不用总是将参数写入计算图的定义, 而可以使用程序参数的方式动态指定参数
def create_dataset_dynamic(filenames):
    # 从TFRecord文件创建数据集，具体文件路径是一个placeholder，稍后再提供具体路径。
    input_files = tf.placeholder(tf.string)
    dataset = tf.data.TFRecordDataset(input_files)
    dataset = dataset.map(parser)

    # 定义遍历dataset的initializable_iterator。
    iterator = dataset.make_initializable_iterator()
    image, label = iterator.get_next()

    with tf.Session() as sess:
        # 首先初始化iterator，并给出input_files的值。
        sess.run(iterator.initializer, feed_dict={input_files: filenames})

        # 遍历所有数据一个epoch。
        # 当遍历结束时，程序会抛出OutOfRangeError。
        while True:
            try:
                x, y = sess.run([image, label])
                # print(x[:5], y) # 输出很多
            except tf.errors.OutOfRangeError:
                break




create_dataset_from_array()

# 创建文本文件作为本例的输入。
with open("./test1.txt", "w") as file:
    file.write("File1, line1.\n")
    file.write("File1, line2.\n")
with open("./test2.txt", "w") as file:
    file.write("File2, line1.\n")
    file.write("File2, line2.\n")
create_dataset_from_txt(["./test1.txt", "./test2.txt"])

create_dataset_from_TFRecord(["../7.1/output.tfrecords"])

create_dataset_dynamic(["../7.1/output.tfrecords"])
