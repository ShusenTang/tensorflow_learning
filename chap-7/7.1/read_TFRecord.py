# 2. 读取TFRecord文件
import tensorflow as tf

# 读取文件。
reader = tf.TFRecordReader()
# 创建一个队列来维护输入文件列表, 7.3.2节会有详细介绍
filename_queue = tf.train.string_input_producer(["output.tfrecords"])
# 从文件中读取出一个样例, 也可以使用read_up_to函数一次性读出多个样例
_, serialized_example = reader.read(filename_queue)

# 解析读取的样例。若需解析多个样例, 则可使用parse_example函数
features = tf.parse_single_example(
    serialized_example,
    features={
        # TensorFlow提供两种不同的属性解析方法. 一种是tf.FixedLenFeature, 这种方法的解析结果为一个tensor;
        # 另一种方法是tf.VarLenFeature,这种方法得到的解析结果是SparseTensor, 用于处理稀疏数据.
        # 这里解析数据的格式要与写入数据的格式一致
        'image_raw': tf.FixedLenFeature([], tf.string),
        'pixels': tf.FixedLenFeature([], tf.int64),
        'label': tf.FixedLenFeature([], tf.int64)
    })

# tf.decode_raw可以将字符串解析成图像对应的像素按钮
images = tf.decode_raw(features['image_raw'], tf.uint8)
labels = tf.cast(features['label'], tf.int32)
pixels = tf.cast(features['pixels'], tf.int32)

sess = tf.Session()

# 启动多线程处理输入数据。7.3节会更加详细介绍多线程处理
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)

for i in range(10):
    image, label, pixel = sess.run([images, labels, pixels])
    print(label)
print("done!")