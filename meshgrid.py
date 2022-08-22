# _*_ coding:utf-8 _*_
# @Time: 2022/8/22 22:12
# @Author: JackZhang9
import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def sincfunc(i,j):
    tmp=tf.sqrt(tf.square(i)+tf.square(j))
    k=tf.sin(tmp)/tmp
    return k

# points=[]  # 采样点
# for i in range(-8,8,100):
#     for j in range(-8,8,100):
#         k=sincfunc(i,j)
#         points.append([i,j,k])

# 在x、y上采样
x=tf.linspace(-8,8,100)  # 在x轴上采样100点
y=tf.linspace(-8,8,100)  # 在y轴上采样100点
# 生成网格点，保存shape为[2,100,100]的张量，返回两个切割后的张量，x、y,shape[100,100]
x,y=tf.meshgrid(x,y)
z=sincfunc(x,y)
fig=plt.figure()
ax=Axes3D(fig)
ax.contour3D(x.numpy(),y.numpy(),z.numpy(),50)
plt.show()
