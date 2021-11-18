#!/usr/bin/env python
# _*_coding: utf-8 _*_
# @Time : 2021/11/18 15:05
# @Author : CN-JackZhang
# @File: 数据类型.py
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import numpy as np

#数据容器
# list,  [1,1.2,'hello',(1,2),layers]
# np.array, [64,224,224,3],不支持gpu计算
# tf.Tensor, 偏重神经网络的计算

# Tensor是什么？
# 0维是标量scalar: 1.1
# 1维是向量vector: [1.1],[1.1,2.2,...]
# 2维是矩阵matrix: [[1.1,2.2],[3.3,4.4],[5.5,6.6]]
# 3维及以上是tensor,rank(秩)>2,即3维起步: [[[]],]
# 在tf里，所有数据都叫tensor,

# 在tf里基本数据类型？
# 跟numpy相似
# int,float,double,
# bool
# string

a = tf.constant(1.,dtype=tf.double)    # 创建一个int数据,这是一个标量
print(a)
b = tf.constant([True,False])       # 这是一个向量，bool类型元素
print(b)
c = tf.constant(['hello world'])      # 这是一个字符串
print(c)


# tf常用属性和方法
# 在cpu环境下创建tensor
with tf.device('cpu'):       # 对于一个tensor，常用属性device,是一个string类型
    d = tf.constant([1])
    print(a.device)     # 返回当前tensor所在设备的名字
with tf.device('gpu'):
    e = tf.range(4)
    print(e.device)

dd = d.gpu()    # 将cpu上的tensor用gpu
print(dd.device)

print(b.numpy())   # 将tensor转化为numpy

print(b.ndim)   # 看tensor维度
print(b.shape)
print(tf.rank(b))   # 看tensor的维度

print(tf.rank(tf.ones([3,4,2])))  # 3维的tensor类型数据
print(tf.ones([3,4,2]))
# 小结：得到维度有b.ndim,得到shape用b.shape返回，得到numpy用b.numpy(),得到设备用b.device。
# cpu/gpu转化用b.cpu()/b.gpu()，b.dtype返回数据类型


# 检测是否是tensor
tf.is_tensor(b)

# numpy转换成tensor
f = np.arange(4)  # numpy类型int64
ff = tf.convert_to_tensor(f,dtype=tf.int32)  # 转换成int32

fff = tf.cast(ff, dtype=tf.double)

# 数据类型转换
g = tf.constant([0,1])
gg = tf.cast(g,dtype=tf.bool)

# tf.Variable()
# 需要优化的
w = tf.Variable(w)   # 自动有可求导的属性,自动记录梯度信息，专门维神经网络的参数设立的类型
w.trainable  # 告诉网络，在向后传播时，对w进行求导，

isinstance(w, tf.Variable)  # 判断
# 一般使用is_tensor或dtype判断类型


# tensor变回numpy,tensor一般运行在gpu上，可能在cpu上控制，b.numpy()

# 如果tensor是标量，可以用int()，或float()得到
h = tf.ones([])
a.numpy()    # s 是一个标量，对于loss常用
int(a)
float(a)














