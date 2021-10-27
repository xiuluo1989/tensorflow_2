import tensorflow as tf

class my_Layer(tf.keras.layers.Layer):

    def __init__(self):
        super(my_Layer, self).__init__()

    def build(self, input_shape):
        '''
        在第一次使用该层的时候调用该部分代码，在这里创建变量可以使得变量的形状自适应输入的形状							#	而不需要使用者额外指定变量形状。9.									#	如果已经可以完全确定变量的形状，也可以在__init__部分创建变量
        :param input_shape: TensorShape	类型对象，提供输入的形状
        :return:
        '''
        self.variable_0 = self.add_weight()
        self.variable_1 = self.add_weight()

    # def call(self, inputs):
    #     # 模型调用的代码（处理输入并返回输出）
    #     return output

class my_Dense(tf.keras.layers.Layer):

    def __init__(self, units):
        super(my_Dense, self).__init__()
        self.units = units

    def build(self, input_shape):
        '''

        :param input_shape: 第一次运行call()时参数inputs的形状
        :return:
        '''
        self.w = self.add_variable(name='w', shape=[input_shape[-1], self.units], initializer=tf.zeros_initializer())
        self.b = self.add_variable(name='b', shape=[self.units], initializer=tf.zeros_initializer())

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

class MeanSquareError(tf.keras.losses.Loss):

    def __init__(self):
        super(MeanSquareError, self).__init__()

    def call(self, y_true, y_pred):
        return tf.reduce_mean(tf.square(y_pred - y_true))

class SparseCategoricalAccuracy(tf.keras.metrics.Metric):

    def __init__(self):
        super(SparseCategoricalAccuracy, self).__init__()
        self.total = self.add_weight(name='total', dtype=tf.int32, initializer=tf.zeros_initializer())
        self.count = self.add_weight(name='count', dtype=tf.int32, initializer=tf.zeros_initializer())

    def update_state(self, y_true, y_pred, sample_weight=None):
        values = tf.cast(tf.equal(y_true, tf.argmax(y_pred, axis=-1, output_type=tf.int32)), tf.int32)
        self.total.assign_add(tf.shape(y_true)[0])
        self.count.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.count / self.total
