#!/usr/bin/env python
# _*_coding: utf-8 _*_
# @Time : 2021/11/17 19:15
# @Author : CN-JackZhang
# @File: 手写数字问题一.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # 减少log信息,一定要放在导入tf库前
import tensorflow as tf
from tensorflow.keras import datasets, layers, optimizers

# 载入mnist数据集
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()  # 载入mnist数据集
x_train = tf.convert_to_tensor(x_train,dtype=tf.float32) / 255.   # numpy类型转化为tensor，每个数据为float32
y_train = tf.convert_to_tensor(y_train,dtype=tf.int32)      # 每个数据我int32类型
y_train = tf.one_hot(y_train,depth=10)  # 10个数字，10维的张量，one_hot编码

train_dataset = tf.data.Dataset.from_tensor_slices((x_train,y_train))   # 把tensor合并成一个训练集
train_dataset = train_dataset.batch(200)  # 一次载入200张图片,一共有60 000张

# 网络结构
model = tf.keras.Sequential([                      # 2层隐藏层全连接神经网络
    layers.Dense(512,activation='relu'),   # 第一层全连接512个神经元节点
    layers.Dense(256,activation='relu'),  # 第二层256
    layers.Dense(10)])                    # 输出层10
optimizer = optimizers.SGD(learning_rate=0.001)  # 随机梯度下降优化，学习率0.001


def train_epoch(epoch):
    for step,(x,y) in enumerate(train_dataset):     # 循环，300次，共有30个epoch，每个epoch有300个step，
        with tf.GradientTape() as tape:
            x = tf.reshape(x,(-1,28*28))   # 把x的[b,28,28]tensor变成[b,784]tensor
            output = model(x)       # 计算输出，根据x[b,784]输出一个y[b,10]
            loss = tf.reduce_sum(tf.square(output-y)) / x.shape[0]    # 计算损失：预测值与真实值的差的平方求和，除以总个数
        grads = tape.gradient(loss,model.trainable_weights)     # 优化并更新权重w1,w2,..,b1,b2..
        optimizer.apply_gradients(zip(grads,model.trainable_weights))
        if step % 100 ==0:
            print(epoch,step,'loss',loss.numpy())

def train():
    for epoch in range(30):     # 对数据集迭代30次,有30个epoch，
        train_epoch(epoch)

if __name__ == '__main__':
    train()
        












































