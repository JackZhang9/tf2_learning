# _*_ coding:utf-8 _*_
# Author:JackZhang9
# Time:2022.8.29 14:40

import tensorflow as tf
import tensorflow.keras as keras
from keras import datasets

# 加载数据集

# boston_housing :回归预测房价
# CIFAR10/100 ：分类图片
# MNIST/Fashion_MNIST :手写数字图片数据集，图片分类
# IMDB ：文本分类
#
# 加载方式：datasets.name.load_data()

# boston数据集加载
(x_b_tra,y_b_tra),(x_b_te,y_b_te)=datasets.boston_housing.load_data()
print(x_b_tra.shape)
# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/boston_housing.npz
# 57026/57026 [==============================] - 0s 2us/step
# (404, 13)

# CIFAR10分类图片
# (x_c_tra,y_c_tra),(x_c_te,y_c_te)=datasets.cifar10.load_data()
# print(x_c_tra.shape)

# mnist数据集加载
(x_m_tra,y_m_tra),(x_m_te,y_m_te)=datasets.mnist.load_data()
print(x_m_tra.shape)
# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
# 11490434/11490434 [==============================] - 2s 0us/step
# (60000, 28, 28)

# IMDB ：文本分类
(x_i_tra,y_i_tra),(x_i_te,y_i_te)=datasets.imdb.load_data()
print(x_i_tra.shape)
# Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
# 17464789/17464789 [==============================] - 1s 0us/step
# (25000,)

# 查看下载好的数据的shape
train_db_b=tf.data.Dataset.from_tensor_slices((x_m_tra,y_m_tra)).batch(128)
for i,(x,y) in enumerate(train_db_b):
    print(i,x.shape,y.shape)

# 部分shape为
# 0 (128, 28, 28) (128,)
# 1 (128, 28, 28) (128,)
# 2 (128, 28, 28) (128,)
# 3 (128, 28, 28) (128,)
# 4 (128, 28, 28) (128,)
# 5 (128, 28, 28) (128,)
# 6 (128, 28, 28) (128,)









