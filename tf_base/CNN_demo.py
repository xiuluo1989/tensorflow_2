import numpy as np
import tensorflow as tf

class MNIST_Loader():

    def __init__(self):
        mnist = tf.keras.datasets.mnist
        (self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
        # MNIST中的图像默认为uint8（0-255的数字）。
        # 以下代码将其归一化到0-1之间的浮点数，并在最后增加一维作为颜色通道
        self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 2]
        self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 2]
        self.train_label = self.train_label.astype(np.int32)  # 60000
        self.test_label = self.test_label.astype(np.int32)  # 10000
        self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

    def get_batch(self, batch_size):
        # 从数据集中随机取出batch_size个元素并返回
        index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
        return self.train_data[index, :], self.train_label[index]

class CNN(tf.keras.Model):

    def __init__(self):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv2D(
            filters=32,  # 卷积神经元(卷积核)数量
            kernel_size=[5, 5],  # 视野大小
            padding='same',  # padding 策略
            activation=tf.nn.relu  # 激活函数
        )
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.conv2 = tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=[5, 5],
            padding='same',
            activation=tf.nn.relu
        )
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2)
        self.flatten = tf.keras.layers.Reshape(target_shape=(7 * 7 * 64,))
        self.dense1 = tf.keras.layers.Dense(units=1024, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(units=10)

    def call(self, inputs):
        x = self.conv1(inputs)  # [batch_size, 28, 28, 32]
        x = self.pool1(x)  # [batch_size, 14, 14, 32]
        x = self.conv2(x)  # [batch_size, 14, 14, 64]
        x = self.pool2(x)  # [batch_size, 7, 7, 64]
        x = self.flatten(x)	 # [batch_size,	7 *	7 * 64]
        x = self.dense1(x)  # [batch_size, 1024]
        x = self.dense2(x)  # [batch_size, 10]

        return tf.nn.softmax(x)

def train(data_loader):
    model = CNN()
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    num_batch = int(data_loader.num_train_data // batch_size * num_epoch)
    for batch_index in range(num_batch):
        X, y = data_loader.get_batch(batch_size)
        with tf.GradientTape() as tape:
            y_pred = model(X)
            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
            loss = tf.reduce_mean(loss)
            print("batch, %d:	loss, %f" % (batch_index,	loss.numpy()))
            grads = tape.gradient(loss, model.variables)
            optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

    return model

def evaluation(model, data_loader, batch_size):
    sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    num_batches = int(data_loader.num_test_data // batch_size)
    for batch_index in range(num_batches):
        start_index, end_index = batch_index * batch_size, (batch_size + 1) * batch_size
        if start_index < end_index:
            y_pred = model.predict(data_loader.test_data[start_index: end_index])

            sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index],
                                                     y_pred=y_pred)
            print("test	accuracy:%f" % sparse_categorical_accuracy.result())

if __name__ == "__main__":
    num_epoch = 5
    batch_size = 32
    learning_rate = 0.001
    data_loader = MNIST_Loader()
    model = train(data_loader)
    evaluation(model, data_loader, batch_size)
    pass
