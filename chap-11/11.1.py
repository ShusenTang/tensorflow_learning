import tensorflow as tf

# 1. 不同的命名空间。
with tf.variable_scope("foo"):
    a = tf.get_variable("bar", [1])
    print(a.name)  # foo/bar:0

with tf.variable_scope("bar"):
    b = tf.get_variable("bar", [1])
    print(b.name)  # bar/bar:0


# 2. tf.Variable和tf.get_variable的区别:唯一区别是使用tf.get_variable函数时
with tf.name_scope("a"):
    a = tf.Variable([1])
    print(a.name)  # a/Variable:0

    # tf.get_variable不受tf.name_scope的影响，所以b并不是在命名空间a下
    b = tf.get_variable("b", [1])
    print(b.name)  # b:0

# 因为tf.get_variable不受tf.name_scope的影响，所以这里视图获取名称为b的变量，但是这个变量已经被声明，所以会报重复声明的错
# with tf.name_scope("b"):
#     b = tf.get_variable("b", [1])


# 3. TensorBoard可以根据命名空间来整理可视化效果图上的节点
with tf.name_scope("input1"):
    input1 = tf.constant([1.0, 2.0, 3.0], name="input2")
with tf.name_scope("input2"):
    input2 = tf.Variable(tf.random_uniform([3]), name="input2")
output = tf.add_n([input1, input2], name="add")

# 定义一个写日志的writer，并将当前的计算图写入日志
writer = tf.summary.FileWriter("log/simple_example-11.1", tf.get_default_graph())
writer.close()
