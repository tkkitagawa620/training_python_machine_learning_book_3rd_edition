from pickletools import optimize
import tensorflow_datasets as tfds
import pathlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os

"""
テンソルを作成するさまざまな方法
"""
np.set_printoptions(precision=3)
a = np.array([[1, 2, 3, 4, 5]], dtype=np.int32)
b = [[6, 7, 8, 9, 10]]
t_a = tf.convert_to_tensor(a)
t_b = tf.convert_to_tensor(b)
# print(t_a)
# print(t_b)

t_ones = tf.ones((2, 3))
# print(t_ones.shape)
# print(t_ones)

tf.random.set_seed(1)
t1 = tf.random.uniform(shape=(5, 2), minval=-1.0, maxval=1.0)
t2 = tf.random.normal(shape=(5, 2), mean=0.0, stddev=1.0)
# print(t1)
# print(t2)

"""
行列のアダマール積
.numpy() と付けるとtensorのオブジェクトではなくnumpy配列で返ってくる。
"""
t3 = tf.multiply(t1, t2).numpy()
# print(t3)

"""
行列の積
一次元のテンソルの転地を得るためには、宣言時に
tensor = [[0,0,0,0,0]]
のように二次元配列にしなければならない。
tensor_1d = [0,0,0,0,0]
print(tf.transpose(tensor_id))   // 同じものが返ってくる
"""
t_ab = tf.linalg.matmul(t_a, t_b, transpose_a=True)
t_ba = tf.linalg.matmul(t_a, t_b, transpose_b=True)
# print(t_ab.numpy())
# print(t_ba.numpy())

"""
データセットを作成する。
"""
a = [1.2, 3.4, 7.5, 4.1, 5.0, 1.0]
ds = tf.data.Dataset.from_tensor_slices(a)
# for item in ds:
# print(item)

"""
データを結合する。
ex. 特徴量データとラベルデータを結合して一つのデータセットとして扱いたいとき。
一つのデータに結合されて、それぞれの要素はタプルで操作できるようになる。
"""
tf.random.set_seed(1)
t_x = tf.random.uniform([4, 3], dtype=np.float32)
t_y = tf.range(4)
ds_x = tf.data.Dataset.from_tensor_slices(t_x)
ds_y = tf.data.Dataset.from_tensor_slices(t_y)
ds_joint = tf.data.Dataset.zip((ds_x, ds_y))
# ds_joint = tf.data.Dataset.from_tensor_slices((t_x, t_y))   # これも同義
# for example in ds_joint:
#     print(' x:', example[0].numpy(), ' y:', example[1].numpy())

"""
全てのデータに一様な変換を行う
"""
ds_trans = ds_joint.map(lambda x, y: (x * 2 - 1.0, y))
# for example in ds_trans:
#     print(' x:', example[0].numpy(), ' y:', example[1].numpy())

"""
データをシャッフルする
"""
ds = ds_joint.shuffle(buffer_size=len(t_x))
# for example in ds:
# print(' x:', example[0].numpy(), ' y:', example[1].numpy())

"""
データセットからバッチを作成する
"""
ds = ds_joint.batch(batch_size=3, drop_remainder=False)
batch_x, batch_y = next(iter(ds))
# print('Batch-x:\n', batch_x.numpy())

"""
画像を読み込む
"""
imgdir_path = pathlib.Path('./python-machine-learning-book-3rd-edition/ch13/cat_dog_images')
file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])
# print(file_list)

"""
tfとmatplotlibを使って画像を可視化する。
"""
# fig = plt.figure(figsize=(10, 5))
# for i, file in enumerate(file_list):
#     img_raw = tf.io.read_file(file)
#     img = tf.image.decode_image(img_raw)
#     ax = fig.add_subplot(2, 3, i + 1)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.imshow(img)
#     ax.set_title(os.path.basename(file), size=15)
# plt.tight_layout()
# plt.show()

"""
ファイル名にラベルが含まれているので、それを抽出する
"""
labels = [1 if 'dog' in os.path.basename(file) else 0 for file in file_list]
# print(labels)

"""
ラベルとファイル名の配列を結合してデータセットを作成する
"""
ds_file_labels = tf.data.Dataset.from_tensor_slices((file_list, labels))
# for item in ds_file_labels:
#     print(item[0].numpy(), item[1].numpy())


"""
画像の読み込みと前処理
"""


def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image /= 255
    return image, label


img_width, img_height = 120, 80
ds_images_labels = ds_file_labels.map(load_and_preprocess)

# fig = plt.figure(figsize=(10, 5))
# for i, example in enumerate(ds_images_labels):
#     ax = fig.add_subplot(2, 3, i + 1)
#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.imshow(example[0])
#     ax.set_title('{}'.format(example[1].numpy()), size=15)
# plt.tight_layout()
# plt.show()

"""
tensorflow-datasetsで提供されている利用可能なデータセットを見てみる
"""
# print('利用可能でデータセットの数:', len(tfds.list_builders()))
# print(tfds.list_builders()[:5])

"""
CelebAデータセットを読み込む
download_and_prepare()メソッドでは読み込めなかったので手動でダウンロードして展開した
"""
celeba_bldr = tfds.builder('celeb_a')
# celeba_bldr.download_and_prepare()
# print(celeba_bldr.info.features)

"""
Keras API を使って(まずは)線形回帰モデルを作成する。
ただし、先にデータを用意しておく
"""
X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1, 2.0, 5.0, 6.3, 6.6, 7.4, 8.0, 9.0])
X_train_norm = (X_train - np.mean(X_train)) / np.std(X_train)
ds_train_orig = tf.data.Dataset.from_tensor_slices((tf.cast(X_train_norm, tf.float32), tf.cast(y_train, tf.float32)))
# plt.plot(X_train, y_train, 'o', markersize=10)
# plt.xlabel('X')
# plt.ylabel('Y')
# plt.show()


class RegressionModel(tf.keras.Model):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.w = tf.Variable(0.0, name='weight')
        self.b = tf.Variable(0.0, name='bias')

    def call(self, x):
        return self.w * x + self.b


"""
作成したモデルを確認
"""
model = RegressionModel()
model.build(input_shape=(None, 1))
# model.summary()

"""
損失関数と学習を行うための関数を定義する
"""


def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss_fn(model(inputs), outputs)
    dW, db = tape.gradient(current_loss, [model.w, model.b])
    model.w.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


"""
定義したモデルや関数を使って実際に回帰分析を行ってみる
"""
tf.random.set_seed(1)
num_epochs = 200
log_steps = 100
learning_rate = 0.001
batch_size = 1
steps_per_epoch = int(np.ceil(len(y_train) / batch_size))

ds_train = ds_train_orig.shuffle(buffer_size=len(y_train))
ds_train = ds_train.repeat(count=None)
ds_train = ds_train.batch(1)
Ws, bs = [], []

# for i, batch in enumerate(ds_train):
#     if i >= steps_per_epoch * num_epochs:
#         break
#     Ws.append(model.w.numpy())
#     bs.append(model.b.numpy())
#     bx, by = batch
#     loss_val = loss_fn(model(bx), by)
#     train(model, bx, by, learning_rate=learning_rate)
#     if i & log_steps == 0:
#         print('Epoch {:4} Step {:2d} Loss {:6.4f}'.format(int(i / steps_per_epoch), i, loss_val))

# print('Final paremeters: ', model.w.numpy(), model.b.numpy())        // Final paremeters:  2.6576622 4.8798566


"""
今度はKeras APIを用いて多層パーセプトロンを構築する

【検証】ニューロンの数を変えてみてモデルの精度がどの程度変化するのか検証した
ニューロン数    Test loss      Test Acc.
--------------------------------------
8             0.109270        0.9700
16            0.090668        0.9700
32            0.076062        0.9800
【結論】ニューロンの数を上げると精度は良くなるが、ニューロンの数と精度の向上は対数的に頭打ちになる

【検証】隠れ層の数を変えてみてモデルの精度がどの程度変化するのか検証
隠れ層の数       パラメータ数       Test loss      Test Acc.
----------------------------------------------------------
2               132(80,51)       0.090668        0.9700
3             403(80,272,51)     0.070546        0.9700
4            675(80,272,272,51)  0.066118        0.9800
【結論】ニューロン数同様に精度は上がるが精度は頭打ちになる。
"""
iris, iris_info = tfds.load('iris', with_info=True)
# print(iris_info)
ds_orig = iris['train']
ds_orig = ds_orig.shuffle(150, reshuffle_each_iteration=False)
ds_train_orig = ds_orig.take(100)
ds_test = ds_orig.take(100)

ds_train_orig = ds_train_orig.map(lambda x: (x['features'], x['label']))
ds_test = ds_test.map(lambda x: (x['features'], x['label']))

iris_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='sigmoid', name='fc1', input_shape=(4,)),
    tf.keras.layers.Dense(16, activation='sigmoid', name='fc2', input_shape=(16,)),
    tf.keras.layers.Dense(16, activation='sigmoid', name='fc3', input_shape=(16,)),
    tf.keras.layers.Dense(3, name='fc4', activation='softmax')
])
# iris_model.summary()

"""
iris_modelをコンパイルする
"""
iris_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

num_epochs = 200
training_size = 100
batch_size = 2
steps_per_epoch = np.ceil(training_size / batch_size)
ds_train = ds_train_orig.shuffle(buffer_size=training_size)
ds_train = ds_train.repeat()
ds_train = ds_train.batch(batch_size=batch_size)
ds_train = ds_train.prefetch(buffer_size=1000)
# history = iris_model.fit(ds_train, epochs=num_epochs, steps_per_epoch=steps_per_epoch, verbose=0)

"""
エポック数と損失率と正答率を描画する
"""
# hist = history.history
# fig = plt.figure(figsize=(12, 5))
# ax = fig.add_subplot(1, 2, 1)
# ax.plot(hist['loss'], lw=3)
# ax.set_title('Training loss', size=15)
# ax.set_xlabel('Epoch', size=15)
# ax.tick_params(axis='both', which='major', labelsize=15)
# ax = fig.add_subplot(1, 2, 2)
# ax.plot(hist['accuracy'], lw=3)
# ax.set_title('Accuracy', size=15)
# ax.set_xlabel('Epoch', size=15)
# ax.tick_params(axis='both', which='major', labelsize=15)
# plt.tight_layout()
# plt.show()


"""
作成したモデルの保存
"""
# iris_model.save('iris-classifier.h5', overwrite=True, include_optimizer=True, save_format='h5')

"""
作成したモデルの読み込み
"""
# iris_model = tf.keras.models.load_model('iris-classifier.h5')
# results = iris_model.evaluate(ds_test.batch(50), verbose=0)
# print('Test loss: {:4f} Test Acc.: {:.4f}'.format(*results))


"""
ロジスティク関数のまとめ
"""


def logistic(z):
    return 1.0 / (1.0 + np.exp(-z))


W = np.array([
    [1.1, 1.2, 0.8, 0.4],
    [0.2, 0.4, 1.0, 0.2],
    [0.6, 1.5, 1.2, 0.7],
])
A = np.array([[1, 0.1, 0.4, 0.6]])
Z = np.dot(W, A[0])
y_probas = logistic(Z)
print('Net input: ', Z)
print('Output units: ', y_probas)

Z_tensor = tf.expand_dims(Z, axis=0)
print(tf.keras.activations.softmax(Z_tensor))
