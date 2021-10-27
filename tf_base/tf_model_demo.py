import numpy as np
import tensorflow as tf
'''
TensorFlow	快速搭建动态模型。
模型的构建：tf.keras.Model and tf.keras.layers
模型的损失函数：tf.keras.losses
模型的优化器：	tf.keras.optimizer
模型的评估：tf.keras.metrics
'''

class my_Model(tf.keras.Model):

    def __init__(self):
        super().__init__()
        # 此处添加初始化代码（包含call方法中会用到的层
        self.layer1 = tf.keras.layers.BuiltInLayer()
        self.layer2 = self.MyCustomLayer()

    def MyCustomLayer(self):
        pass

    def call(self, input):
        # 此处添加模型调用的代码(处理输入并返回输出)
        x = self.layer1(input)
        output = self.layer2(x)

        return output

class my_Linear(tf.keras.Model):

    def __init__(self):
        super(my_Linear, self).__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input):
        return self.dense(input)

def use_My_Linear(X, y):
    model = my_Linear()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)
    for i in range(100):
        with tf.GradientTape() as tape:
            y_pred = model(X)  # 调用模型	y_pred = model(X) 而不是显式写出 y_pred = a * X + b
            loss = tf.reduce_mean(tf.square(y_pred - y))
            grads = tape.gradient(loss, model.variables)  #	使用	model.variables	这一属性直接获得模型中的所有变量
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    return model.variables

if __name__ == "__main__":
    X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[10.0], [20.0]])

    variables = use_My_Linear(X, y)
    print(variables)
    pass
