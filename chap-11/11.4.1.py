import matplotlib.pyplot as plt
import PIL
import numpy as np
import os
from tensorflow.examples.tutorials.mnist import input_data

# PROJECTOR要求用户准备一个sprite图像(即由一组图片组合成的一大张图片)和一个tsv文件给每张图片对应的真实标签
LOG_DIR = "log/simple_example-11.4.1"
SPRITE_FILE = 'mnist_sprite.png'
META_FIEL = "mnist_meta.tsv"


# 使用给出的mnist图片列表生成sprite图像
def create_sprite_image(images):
    """Returns a sprite image consisting of images passed as argument. Images should be count x width x height"""
    if isinstance(images, list):
        images = np.array(images)
    img_h = images.shape[1]
    img_w = images.shape[2]
    # sprite图像就是用很多小图像拼接成一个正方形的矩阵，此正方形的边长就是sqrt(小图像数量)，注意向上取整
    n_plots = int(np.ceil(np.sqrt(images.shape[0])))

    spriteimage = np.ones((img_h * n_plots, img_w * n_plots)) # 初始化

    for i in range(n_plots):
        for j in range(n_plots):
            this_img_index = i * n_plots + j  # 计算当前图片编号
            if this_img_index < images.shape[0]:
                this_img = images[this_img_index]
                spriteimage[i * img_h:(i + 1) * img_h, j * img_w:(j + 1) * img_w] = this_img
    return spriteimage


# 这里one_hot=False，于是得到的label不是二进制vector而是一个数字，表示当前图片表示的数字
mnist = input_data.read_data_sets("../datasets/MNIST_data", one_hot=False)

# 生成sprite图像
to_visualise = 1 - np.reshape(mnist.test.images, (-1, 28, 28))  # 图片背景是黑色的，用1-...调换一下使背景为白色
sprite_image = create_sprite_image(to_visualise)

# 创建日志文件夹
folder = os.path.exists(LOG_DIR)
if not folder:  # 判断是否存在文件夹如果不存在则创建为文件夹
    os.makedirs(LOG_DIR)

# 将sprite图像放到相应日志目录下
path_for_mnist_sprites = os.path.join(LOG_DIR, SPRITE_FILE)
plt.imsave(path_for_mnist_sprites, sprite_image, cmap='gray')  # cmap='gray' -> 灰度图

path_for_mnist_metadata = os.path.join(LOG_DIR, META_FIEL)
# 生成每张图片对应标签文件并写到对应的日志目录下
with open(path_for_mnist_metadata, 'w') as f:
    f.write("Index\tLabel\n")
    for index, label in enumerate(mnist.test.labels):
        f.write("%d\t%d\n" % (index, label))
