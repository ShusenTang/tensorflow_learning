import tensorflow as tf

from numpy.random import RandomState

batch_size = 8

x = tf.placeholder(tf.float32, shape=(None, 2), name='x-input')
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')

#定义神经网络的前向传播过程:简单的加权和
w1 = tf.Variable(tf.random_normal([2,1], stddev=1, seed=1))
y = tf.matmul(x,w1)

#定义损失函数和反向传播算法
loss_less = 10
loss_more = 1
loss = tf.reduce_sum(tf.where(tf.greater(y, y_),(y - y_) * loss_more, (y_ - y) * loss_less)) # 训练结果倾向于y>y_
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

#生成模拟数据
sample_num = 256
rdm = RandomState(1)
X = rdm.rand(sample_num, 2)
Y = [[x1 + x2 + rdm.rand()/10.0 - 0.05] for (x1, x2) in X] # y = x1 + x2 + (-0.05~0.05)

# 创建会话运行程序
with tf.Session() as sess:
    init_op = tf.global_variables_initializer() # 初试所有变量
    sess.run(init_op)

    # 输出目前（未经训练）的参数取值。
    print("训练前参数:")
    print("w1:\n", sess.run(w1))
    print("\n")

    # 训练模型
    STEPS = 10000
    for i in range(STEPS): # 迭代STEPS次
        start = (i * batch_size) % sample_num # 每次选取batch_size个样本进行训练
        end = min(start + batch_size, sample_num)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})



    # 输出训练后的参数取值。
    print("训练后参数:")
    print("w1:\n", sess.run(w1))
    print("可以看出预测值倾向于大于实际值\n")



