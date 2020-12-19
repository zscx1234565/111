import tensorflow as tf
print(tf.__version__)
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Reserve 10,000 samples for validation  拆分成训练集和验证集
x_val = x_train[-10000:]
y_val = y_train[-10000:]


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

hm = tf.keras.models.load_model("mnist_model.h5")
hm.evaluate(x_val, y_val)