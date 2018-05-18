"""
将图片转换为tfrecords
"""
import glob
'''
glob用于简单的文件路径名匹配，"*"匹配0个或多个字符；"?"匹配单个字符；"[]"匹配指定范围内的字符，如：[0-9]匹配数字

glob.glob返回所有匹配的文件路径列表
#例子：获取当前目录下的所有图片名: print glob.glob(r"./*.jpg")  

glob.iglob获取一个可编历对象
例子： 
f = glob.iglob(r'../*.py')   
print f #<generator object iglob at 0x00B9FF80>  
for py in f:  
    print py  
'''
import os.path
import random
import tensorflow as tf
import numpy as np
from PIL import Image

# 原始输入数据的目录，这个目录下有5个子目录，每个子目录底下保存这属于该
# 类别的所有图片。
INPUT_DATA = '../../datasets/flower_photos'
OUTPUT_PATH = '../../datasets/flower_photos/tfrecords'

# 测试数据和验证数据比例。
VALIDATION_PERCENTAGE = 20
TRAIN_SHARDS = 8
VALID_SHARDS = 2
IMAGE_SIZE = 299


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
        'label': _int64_feature(label),
        'image_raw': _bytes_feature(image_raw)
    }))
    return example


def create_image_lists(dir_name):
    """返回图片文件名及对应label"""
    sub_dirs = [x[0] for x in os.walk(dir_name)]
    # for x in os.walk(INPUT_DATA): print(x[0])  # 此方法打印INPUT_DATA及此目录下所有文件夹的文件名

    image_filenames = []
    labels = []
    current_label = 0

    # 读取所有的子目录。
    for sub_dir in sub_dirs:
        if sub_dir == INPUT_DATA:
            continue
        # print(sub_dir)

        # 获取一个子目录中所有的图片文件。
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        dir_name = os.path.basename(sub_dir)
        # print(dir_name)  # roses sunflowers daisy dandelion tulips
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            current_image_filenames = glob.glob(file_glob)
            current_labels = [current_label for _ in range(len(current_image_filenames))]
            image_filenames.extend(current_image_filenames)
            labels.extend(current_labels)

        current_label += 1
    assert len(image_filenames) == len(labels)
    #print(len(image_filenames), len(labels))

    # 随机打乱顺序
    shuffled_index = list(range(len(image_filenames)))
    random.seed(1234)
    random.shuffle(shuffled_index)
    image_filenames = [image_filenames[i] for i in shuffled_index]
    labels = [labels[i] for i in shuffled_index]

    assert len(image_filenames) == len(labels)

    return image_filenames, labels


def process_image_files(name, filenames, labels, shards, output_dir = OUTPUT_PATH):  # name == "train" or "valid"
    print('There are %d image in %s dataset.' % (len(filenames), name))

    # 将图片拆分成shards个部分
    spacing = np.linspace(0, len(filenames), shards + 1).astype(np.int)
    ranges = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])
    # print(ranges)  # 例 [[0, 100], [100, 200], [200, 300], [300, 400], [400, 500], [500, 600], [600, 700], [700, 800]]

    output_dir = output_dir + "/" + name
    for i in range(shards):
        print("Process image %d-%d ......" % (ranges[i][0], ranges[i][1]))
        output_file = ('%s/%s_data.tfrecords_%.2d_of_%.2d' % (output_dir, name, i, shards))
        # print(output_file)
        writer = tf.python_io.TFRecordWriter(output_file)
        for index in range(ranges[i][0], ranges[i][1]):
            image_path = filenames[index]
            if not os.path.exists(image_path):
                print("File %s is not exist!!!" % image_path)
                continue

            with Image.open(image_path) as img:
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    print(image_path + " is not RGB mode, transform successfully!")
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
                img_raw = img.tobytes()  # 将图片转化为二进制格式
                example = tf.train.Example(features=tf.train.Features(feature={
                    "label": _int64_feature(labels[index]),
                    'img_raw': _bytes_feature(img_raw)
                }))  # example对象对label和image数据进行封装
                writer.write(example.SerializeToString())  # 序列化为字符串
        writer.close()
        print("Save " + output_file + " successfully!")
    print("Process %s images to tfrecords done!" % name)


def process_all_images(valid_percentage):
    image_filenames, labels = create_image_lists(INPUT_DATA)
    valid_percentage = valid_percentage * 0.01

    valid_image_filenames = image_filenames[: int(len(image_filenames) * valid_percentage )]
    valid_labels = labels[: int(len(image_filenames) * valid_percentage)]

    train_image_filenames = image_filenames[int(len(image_filenames) * valid_percentage) :]
    train_labels = labels[int(len(image_filenames) * valid_percentage) :]

    process_image_files("train", train_image_filenames, train_labels, TRAIN_SHARDS)
    process_image_files("valid", valid_image_filenames, valid_labels, VALID_SHARDS)


def main():
    process_all_images(VALIDATION_PERCENTAGE)
    # 通过numpy格式保存处理后的数据。
    print("done!")


if __name__ == '__main__':
    main()
