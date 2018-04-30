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
import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile # 类似于python文件操作的API,可以对本地和Google云文件进行操作,
# 如果所有文件都放在本地，直接使用python提供的常规文本操作就行了不用使用gfile

# 原始输入数据的目录，这个目录下有5个子目录，每个子目录底下保存这属于该
# 类别的所有图片。
INPUT_DATA = '../../datasets/flower_photos'
# 输出文件地址。我们将整理后的图片数据通过numpy的格式保存, 第七章有更详细的数据预处理介绍，这里先用numpy存储
OUTPUT_FILE = '../../datasets/flower_photos/flower_processed_data.npy'

# 测试数据和验证数据比例。
VALIDATION_PERCENTAGE = 10
TEST_PERCENTAGE = 10


# 读取数据并将数据分割成训练数据、验证数据和测试数据。
def create_image_lists(sess, testing_percentage, validation_percentage):
    sub_dirs = [x[0] for x in os.walk(INPUT_DATA)]
    #for x in os.walk(INPUT_DATA): print(x[0])  # 此方法打印INPUT_DATA及此目录下所有文件夹的文件名

    # 初始化各个数据集。
    training_images = []
    training_labels = []
    testing_images = []
    testing_labels = []
    validation_images = []
    validation_labels = []
    current_label = 0

    # 读取所有的子目录。
    for sub_dir in sub_dirs:
        if sub_dir == INPUT_DATA:
            # print(sub_dir)
            continue

        # 获取一个子目录中所有的图片文件。
        extensions = ['jpg', 'jpeg', 'JPG', 'JPEG']
        file_list = []
        dir_name = os.path.basename(sub_dir)
        # print(dir_name)  # roses sunflowers daisy dandelion tulips
        for extension in extensions:
            file_glob = os.path.join(INPUT_DATA, dir_name, '*.' + extension)
            file_list.extend(glob.glob(file_glob))
        if not file_list: continue
        print("processing:", dir_name)

        i = 0
        # 处理图片数据。
        for file_name in file_list:
            i += 1
            # 读取并解析图片，将图片转化为299*299以方便inception-v3模型来处理。
            # FastGFile快速获取文本操作句柄，类似于python提供的文本操作open()函数，将会返回一个文本操作句柄。
            image_raw_data = gfile.FastGFile(file_name, 'rb').read()

            image = tf.image.decode_jpeg(image_raw_data) # Decode a JPEG-encoded image to a tensor.
            if image.dtype != tf.float32:
                image = tf.image.convert_image_dtype(image, dtype=tf.float32)
            image = tf.image.resize_images(image, [299, 299]) # 使用指定方法改变图片大小，如双线性插值、最近邻插值等
            image_value = sess.run(image)

            # 随机划分数据聚。
            chance = np.random.randint(100)  # 返回0-99随机整数
            if chance < validation_percentage:
                validation_images.append(image_value)
                validation_labels.append(current_label)
            elif chance < (testing_percentage + validation_percentage):
                testing_images.append(image_value)
                testing_labels.append(current_label)
            else:
                training_images.append(image_value)
                training_labels.append(current_label)
            if i % 200 == 0:
                print(i, "images processed.")
        current_label += 1

    # 将训练数据随机打乱以获得更好的训练效果。
    state = np.random.get_state()
    np.random.shuffle(training_images)
    np.random.set_state(state)
    np.random.shuffle(training_labels)

    return np.asarray([training_images, training_labels,
                       validation_images, validation_labels,
                       testing_images, testing_labels])

def main():
    with tf.Session() as sess:
        processed_data = create_image_lists(sess, TEST_PERCENTAGE, VALIDATION_PERCENTAGE)
        # 通过numpy格式保存处理后的数据。
        np.save(OUTPUT_FILE, processed_data)


if __name__ == '__main__':
    main()
