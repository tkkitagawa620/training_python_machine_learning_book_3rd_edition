import tensorflow_datasets as tfds
from mlxtend.plotting import plot_decision_regions
from matplotlib import markers
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

"""
KerasのSequentialクラスを使って、2つの全結合層からなるモデルを構築する
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=16, activation='relu'))
model.add(tf.keras.layers.Dense(units=32, activation='relu'))
model.build(input_shape=(None, 4))
# model.summary()

# for v in model.variables:
#     print('{:20s}'.format(v.name), v.trainable, v.shape)

"""
各層の細かな設定を行っていく。
1層目:
  ニューロン数:16
  活性化関数:ReLu
  カーネルの初期化:Glorot初期化
  バイアスの初期化:定数を設定
2層目:
  ニューロン数:32
  活性化関数:シグモイド(ロジスティク関数)
  正則化: L1(Lasso回帰)

Glorot初期化
  重みを一様分布の乱数で初期化したい場合、一様分布の区間は次のように設定することで
  訓練の効率化と特定の層が置き去りにされる現象を防ぐ。
    W ~ Uniform(-srqt(6)/sqrt(n_in+n_out), srqt(6)/sqrt(n_in+n_out))
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(
    units=16,
    activation=tf.keras.activations.relu,
    kernel_initializer=tf.keras.initializers.glorot_uniform(),
    bias_initializer=tf.keras.initializers.Constant(2.0)
))
model.add(tf.keras.layers.Dense(
    units=32,
    activation=tf.keras.activations.sigmoid,
    kernel_regularizer=tf.keras.regularizers.l1
))
"""
モデルをコンパイルする際にオプティマイザ、損失関数、評価指標を設定している。
"""
model.compile(
    optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[
        tf.keras.metrics.Accuracy(),
        tf.keras.metrics.Precision(),
        tf.keras.metrics.Recall()
    ]
)

"""
XOR問題を解く
"""
tf.random.set_seed(1)
np.random.seed(1)
x = np.random.uniform(low=-1, high=1, size=(200, 2))
y = np.ones(len(x))
y[x[:, 0] * x[:, 1] < 0] = 0
x_train = x[:100, :]
y_train = y[:100]
x_valid = x[:100, :]
y_valid = y[:100]
# fig = plt.figure(figsize=(6, 6))
# plt.plot(x[y == 0, 0], x[y == 0, 1], 'o', alpha=0.75, markersize=10)
# plt.plot(x[y == 1, 0], x[y == 1, 1], '<', alpha=0.75, markersize=10)
# plt.xlabel(r'$x_1$', size=15)
# plt.xlabel(r'$x_2$', size=15)
# plt.show()

"""
層が多くなる→ニューロンの数が増える→パラメータが増える
パラメータが増えるとより複雑な関数に適合することができるが...
    →訓練が難しくなる(過学習に陥りやすくなる)
...よって、原則として単層ニューラルネットワークといった単純なモデルから始めるのが良い
"""
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=4, input_shape=(2,), activation='relu'))
model.add(tf.keras.layers.Dense(units=4, activation='relu'))
model.add(tf.keras.layers.Dense(units=4, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
# model.summary()
model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)
# hist = model.fit(
#     x_train,
#     y_train,
#     validation_data=(x_valid, y_valid),
#     epochs=200,
#     batch_size=2,
#     verbose=0
# )
# model.save(
#     'xor-classifier.h5',
#     overwrite=True,
#     include_optimizer=True,
#     save_format='h5'
# )
# model = tf.keras.models.load_model('xor-classifier.h5')
# ds_valid = tf.data.Dataset.from_tensor_slices((x_valid, y_valid))
# results = model.evaluate(ds_valid.batch(50), verbose=0)
# print('Test loss: {:4f} Test Acc.: {:.4f}'.format(*results))

"""
KerasのFunctional APIを使って同等のモデルを構築する
ここでは関数型のアプローチを用いて実装する。
"""
tf.random.set_seed(1)
inputs = tf.keras.Input(shape=(2,))
h1 = tf.keras.layers.Dense(units=4, activation='relu')(inputs)
h2 = tf.keras.layers.Dense(units=4, activation='relu')(h1)
h3 = tf.keras.layers.Dense(units=4, activation='relu')(h2)
outputs = tf.keras.layers.Dense(units=1, activation='sigmoid')(h3)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# model.summary()

"""
tf.keras.Modelのサブクラス化による実装方法
"""


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hidden_1 = tf.keras.layers.Dense(units=4, activation='relu')
        self.hidden_2 = tf.keras.layers.Dense(units=4, activation='relu')
        self.hidden_3 = tf.keras.layers.Dense(units=4, activation='relu')
        self.output_layer = tf.keras.layers.Dense(units=1, activation='sigmoid')

    def call(self, inputs):
        h = self.hidden_1(inputs)
        h = self.hidden_2(h)
        h = self.hidden_3(h)
        return self.output_layer(h)


model = MyModel()
model.build(input_shape=(None, 2))
# model.summary()
model.compile(
    optimizer=tf.keras.optimizers.SGD(),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[tf.keras.metrics.BinaryAccuracy()]
)
# hist = model.fit(
#     x_train,
#     y_train,
#     validation_data=(x_valid, y_valid),
#     epochs=200,
#     batch_size=2,
#     verbose=0
# )


"""
Kerasのカスタム層の作成方法
"""


class NoisyLinear(tf.keras.layers.Layer):
    def __init__(self, output_dim, noise_stddev=0.1, **kwargs):
        self.output_dim = output_dim
        self.noise_stddev = noise_stddev
        super(NoisyLinear, self).__init__(**kwargs)

    def build(self, input_shape):
        self.w = self.add_weight(name='weights',
                                 shape=(input_shape[1], self.output_dim),
                                 initializer='random_normal',
                                 trainable=True)

        self.b = self.add_weight(shape=(self.output_dim,),
                                 initializer='zeros',
                                 trainable=True)

    def call(self, inputs, training=False):
        if training:
            batch = tf.shape(inputs)[0]
            dim = tf.shape(inputs)[1]
            noise = tf.random.normal(shape=(batch, dim),
                                     mean=0.0,
                                     stddev=self.noise_stddev)

            noisy_inputs = tf.add(inputs, noise)
        else:
            noisy_inputs = inputs
        z = tf.matmul(noisy_inputs, self.w) + self.b
        return tf.keras.activations.relu(z)

    def get_config(self):
        config = super(NoisyLinear, self).get_config()
        config.update({'output_dim': self.output_dim,
                       'noise_stddev': self.noise_stddev})
        return config


tf.random.set_seed(1)
model = tf.keras.Sequential([
    NoisyLinear(4, noise_stddev=0.1),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=4, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid'),
])
model.build(input_shape=(None, 2))
# model.summary()
# model.compile(
#     optimizer=tf.keras.optimizers.SGD(),
#     loss=tf.keras.losses.BinaryCrossentropy(),
#     metrics=[tf.keras.metrics.BinaryAccuracy()]
# )
# hist = model.fit(
#     x_train,
#     y_train,
#     validation_data=(x_valid, y_valid),
#     epochs=200,
#     batch_size=2,
#     verbose=0
# )

"""
mlxtendを使った可視化

【結果】訓練データの正答率より検証データの正答率の方が良くなっている。
参照：./random/ch_xor_custom_layer.png
これはモデルの汎化性能を評価していいのか、それとも別の解釈が必要なのかわからない。
"""
# history = hist.history
#
# fig = plt.figure(figsize=(16, 4))
# ax = fig.add_subplot(1, 3, 1)
# plt.plot(history['loss'], lw=4)
# plt.plot(history['val_loss'], lw=4)
# plt.legend(['Train loss', 'Validation loss'], fontsize=15)
# ax.set_xlabel('Epochs', size=15)

# ax = fig.add_subplot(1, 3, 2)
# plt.plot(history['binary_accuracy'], lw=4)
# plt.plot(history['val_binary_accuracy'], lw=4)
# plt.legend(['Train Acc.', 'Validation Acc.'], fontsize=15)
# ax.set_xlabel('Epochs', size=15)

# ax = fig.add_subplot(1, 3, 3)
# plot_decision_regions(X=x_valid, y=y_valid.astype(np.integer),
#                       clf=model)
# ax.set_xlabel(r'$x_1$', size=15)
# ax.xaxis.set_label_coords(1, -0.025)
# ax.set_ylabel(r'$x_2$', size=15)
# ax.yaxis.set_label_coords(-0.025, 1)
# plt.show()

"""
Estimators を使ってMNISTの手書き文字を分類する
"""
BUFFER_SIZE = 10000
BATCH_SIZE = 64
NUM_EPOCHS = 20
steps_per_epoch = np.ceil(60000 / BATCH_SIZE)


def preprocess(item):
    image = item['image']
    label = item['label']
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.reshape(image, (-1,))
    return {'image-pixels': image}, label[..., tf.newaxis]


def train_input_fn():
    datasets = tfds.load(name='mnist')
    mnist_train = datasets['train']
    dataset = mnist_train.map(preprocess)
    dataset = dataset.shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE)
    return dataset.repeat()


def eval_input_fn():
    dataset = tfds.load(name='mnist')
    mnist_test = dataset['test']
    dataset = mnist_test.map(preprocess).batch(BATCH_SIZE)
    return dataset


image_feature_column = tf.feature_column.numeric_column(key='image-pixels', shape=(28 * 28))
dnn_classifier = tf.estimator.DNNClassifier(
    feature_columns=[image_feature_column],
    hidden_units=[32, 16],
    n_classes=10,
    model_dir='models/mnist-dnn/'
)
dnn_classifier.train(input_fn=train_input_fn, steps=NUM_EPOCHS * steps_per_epoch)
eval_result = dnn_classifier.evaluate(input_fn=eval_input_fn)
print(eval_result)
