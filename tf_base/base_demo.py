import numpy as np
import tensorflow as tf

def demo_01():
    random_float = tf.random.uniform(shape=())  # 定义随机数(标量)
    zero_vector = tf.zeros(shape=(2))  # 2个元素的零向量
    # 两个常量矩阵
    A = tf.constant([[1, 2], [3, 5]])
    B = tf.constant([[4, 6], [7, 8]])
    print(B.numpy())  # 转为 numpy 数组
    print(B.dtype)
    print(B.shape)

def tf_cal_demo():
    A = tf.constant([[1, 2], [3, 5]])
    B = tf.constant([[4, 6], [7, 8]])
    C = tf.add(A, B)  # 矩阵和
    D = tf.matmul(A, B)  # 矩阵乘积, 点乘
    print(C)
    print(D)

def auto_gradient():  # 自求导机制
    # 计算y = x^2 在 x=3 时的导数
    x = tf.Variable(initial_value=3, dtype=tf.float32)  # 自求导是变量类型需要是float类型
    # 在 tf.GradientTape()的上下文内, 所有计算步骤都会被记录以用于求导
    with tf.GradientTape() as tape:
        y = tf.square(x)
        y_grad = tape.gradient(y, x)  # 计算y关于x的导数
        print([y, y_grad])

def auto_partial_derivative():
    '''
    计算函数 L(w, b) = ||X * w + b - y|| ^ 2
    在 w = (1, 2)T, b = 1时对 w, b 的偏导, X = [[1, 2], [3, 4]], y = [1, 2]
    :return:
    '''
    X = tf.constant([[1, 2], [3, 4]], dtype=tf.float32)
    y = tf.constant([1, 2], dtype=tf.float32)
    w = tf.Variable(initial_value=[[1.], [2.]])
    b = tf.Variable(initial_value=1.)

    with tf.GradientTape() as tape:
        L = 0.5 * tf.reduce_sum(tf.square(tf.matmul(X, w) + b - y))
        w_grad, b_grad = tape.gradient(L, [w, b])  # 计算L(w, b)关于 w,b 的偏导数
        print(L.numpy(), ";", w_grad.numpy(), ";", b_grad.numpy())

def numpy_partial_derivative(X, y):
    a, b = 0, 0
    num_epoch = 10000
    learning_rate = 1e-3
    for e in range(num_epoch):
        y_pred = a * X + b  # 手动计算损失函数关于自变量(模型参数)的梯度
        grad_a, grad_b = (y_pred - y).dot(X), (y_pred - y).sum()
        a, b = a - learning_rate * grad_a, b - learning_rate * grad_b  # 更新参数

    return a, b

def tf_partial_derivative(X, y):
    '''
    tape.gradient(ys,	xs)	自动计算梯度；
    optimizer.apply_gradients(grads_and_vars) 自动更新模型参数。
    :param X:
    :param y:
    :return:
    '''
    x = tf.constant(X)
    y = tf.constant(y)

    a = tf.Variable(initial_value=0.)
    b = tf.Variable(initial_value=0.)
    variables = [a, b]

    num_epoch = 10000
    optimizer = tf.keras.optimizers.SGD(learning_rate=1e-3)
    for e in range(num_epoch):
        with tf.GradientTape() as tape:  # 使用tf.GradientTape()记录损失函数的梯度信息
            y_pred = a * X + b
            loss = 0.5 * tf.reduce_sum(tf.square(y_pred - y))

        '''
        TensorFlow自动计算损失函数关于自变量(模型参数)的梯度
        apply_gradients需要传入Python列表（List），列表中的每个元素是一个（变量的偏导数，变量）对。
        这里是[(grad_a, a),	(grad_b, b)]；通过grads=tape.gradient(loss,variables)求出tape中记录的 loss 关于variables=[a,b]		
        中每个变量的偏导数，就是	grads=[grad_a,	grad_b]，
        用Python的	zip()函数将grads=[grad_a, grad_b]和variables=[a,	b]拼装在一起，就可以组合出所需的参数了。
        '''
        grads = tape.gradient(loss, variables)
        optimizer.apply_gradients(grads_and_vars=zip(grads, variables))  # TensorFlow自动根据梯度更新参数

    return a, b

def tf_regression_demo():
    X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
    y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

    # 归一化
    x = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
    y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())

    # 用梯度下降求线性模型参数 a,b 的值; minL(a, b) = sum(a * x + b - y) ^ 2
    # a, b = numpy_partial_derivative(x, y)
    a, b = tf_partial_derivative(x, y)
    pass

if __name__ == "__main__":
    # demo_01()
    # tf_cal_demo()
    # auto_gradient()
    # auto_partial_derivative()
    tf_regression_demo()
    pass
