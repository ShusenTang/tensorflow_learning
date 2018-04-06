'''
定义前向传播过程和网络参数
'''

import tensorflow as tf

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500

def get_weight_variable(shape, regularizer):
    '''
    使用tf.get_variable获取变量：在训练神经网络的时候会创建这些变量，在测试的时候回通过保存的模型加载这些变量
    '''
    weights = tf.get_variable("weights", shape, initializer=tf.truncated_normal_initializer(stddev=0.1))
    if regularizer is not None: tf.add_to_collection('losses', regularizer(weights)) # 将正则化损失加入loss
    return weights


def inference(input_tensor, regularizer):
    '''
    定义神经网络前向传播过程
    '''
    with tf.variable_scope('layer1'):
        # 声明第一层神经网络的变量并完成前向传播的过程
        weights = get_weight_variable([INPUT_NODE, LAYER1_NODE], regularizer)
        biases = tf.get_variable("biases", [LAYER1_NODE], initializer=tf.constant_initializer(0.0))
        layer1 = tf.nn.relu(tf.matmul(input_tensor, weights) + biases)

    with tf.variable_scope('layer2'):
        # 第二层
        weights = get_weight_variable([LAYER1_NODE, OUTPUT_NODE], regularizer)
        biases = tf.get_variable("biases", [OUTPUT_NODE], initializer=tf.constant_initializer(0.0))
        layer2 = tf.matmul(layer1, weights) + biases

    return layer2
