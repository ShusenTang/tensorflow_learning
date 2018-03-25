import tensorflow as tf

from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2,3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3,1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input') # 一维上使用none可以方便使用不同的batch大小
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

#定义神经网络的前向传播过程
a = tf.matmul(x,w1)
y = tf.matmul(a,w2)

#定义损失函数和反向传播算法
y = tf.sigmoid(y)
# tf.clip_by_value(y, 1e-10, 1.0)意思是将y的值限制在1e-10~1.0之间以避免一些运算错误，
# 符号*是矩阵元素的乘法而不是矩阵乘法(tf.matmul)
# reduce_mean是对矩阵的所有元素求平均
cross_entropy = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(y, 1e-10, 1.0)) + (1-y_)*tf.log(tf.clip_by_value(1-y, 1e-10, 1.0)))
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

#生成模拟数据
sample_num = 256
rdm = RandomState(1)
X = rdm.rand(sample_num, 2)
Y = [[int(x1+x2 < 1)] for (x1, x2) in X] # x1+x2<1的样本被认为是正样本

# 创建会话运行程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer() # 初试所有变量
    sess.run(init_op)

    # 输出目前（未经训练）的参数取值。
    print("训练前参数:")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    print("\n")

    # 训练模型
    STEPS = 10000
    for i in range(STEPS): # 迭代STEPS次
        start = (i * batch_size) % sample_num # 每次选取batch_size个样本进行训练
        end = start + batch_size
        sess.run([train_step, y, y_], feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0: # 每隔一段时间计算一次交叉熵
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all data is %g" % (i, total_cross_entropy))

    # 输出训练后的参数取值。
    print("\n训练后参数:")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
    print("\n")



