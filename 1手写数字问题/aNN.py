 # _*_ coding:utf-8 _*_
 # @Time: 2022/8/22 16:44
 # @Author: JackZhang9

 # 构建一个三层神经网络1:784,2.256,3.128,4.10，mnist手写字模型
import tensorflow as tf
import tensorflow.keras as keras   # 注意keras版本与tf版本一致

# 导入minst数据集x_train.shape,y_train.shape:(60000, 28, 28) (60000,)
(x_train, y_train), (x_test, y_test)=keras.datasets.mnist.load_data()
x_train=tf.convert_to_tensor(x_train,dtype=tf.float32)  # 转换成float32数值类型
y_train=tf.convert_to_tensor(y_train,dtype=tf.int32)

# 创建数据集，取一批512个样本，构建一个迭代器
train_xy=tf.data.Dataset.from_tensor_slices((x_train,y_train)).batch(128)
train_iter=iter(train_xy)

# 权重w、b初始化
# 1.第一层权重创建w,b
w1=tf.Variable(tf.random.truncated_normal([784,256],0.,0.01,seed=10))
b1=tf.Variable(tf.zeros([256]))

# 2.第二层w,b
w2=tf.Variable(tf.random.truncated_normal([256,128],0.,0.01,seed=10))
b2=tf.Variable(tf.zeros([128]))

# 3.第三层w,b
w3=tf.Variable(tf.random.normal([128,10],0.,0.01,seed=10))
b3=tf.Variable(tf.zeros([10]))

# 学习率/步长
lr=0.001

def forwardtensor(x_train):
    # 前向传播过程
    # 第一层,矩阵相乘，放入激活函数relu
    layer1=tf.matmul(x_train,w1)+b1
    layer1=tf.nn.relu(layer1)
    # 第二层,矩阵相乘，放入激活函数relu
    layer2=tf.matmul(layer1,w2)+b2
    layer2=tf.nn.relu(layer2)
    # 第三层,矩阵相乘，放入激活函数relu
    out_layer=tf.matmul(layer2,w3)+b3
    return out_layer

def computeloss(out_layer,y_train_onehot):
    # 计算损失函数loss(w,b),采用均方误差损失函数，mse
    loss=tf.reduce_mean(tf.square(out_layer-y_train_onehot))
    return loss

def grdientcompute():
    for epoch in range(100):   # 整个数据集训练次数
        for step,(x,y) in enumerate(train_xy):   # batch，一次取多少数据集
            # 改变x_train维度，视图,(60000, 784)
            x = tf.reshape(x, [-1, 28 * 28])
            # print(x.shape)
            # y_train进行离散化处理，one-hot编码
            y_train_onehot = tf.one_hot(y, 10)

            with tf.GradientTape() as tape:
                # 梯度计算,loss(w,b)的梯度
                out_layer=forwardtensor(x)    # 前向传播
                # print('out',out_layer)
                loss=computeloss(out_layer,y_train_onehot)    # 计算损失函数
                # print('loss',loss)
            # 自动计算梯度,梯度值存于列表
            auto_gradient=tape.gradient(loss,[w1,b1,w2,b2,w3,b3])
            # 梯度下降，更新参数的梯度,assign_sub类似于:a=a-1;
            w1.assign_sub(lr * auto_gradient[0])
            b1.assign_sub(lr * auto_gradient[1])
            w2.assign_sub(lr * auto_gradient[2])
            b2.assign_sub(lr * auto_gradient[3])
            w3.assign_sub(lr * auto_gradient[4])
            b3.assign_sub(lr * auto_gradient[5])
            if step%100 == 0:
                print(epoch,step,'====>',float(loss))

if __name__ == '__main__':
    grdientcompute()

# 第100次循环结果
# 99 0 ====> 0.007306714542210102
# 99 100 ====> 0.009887048043310642
# 99 200 ====> 0.009259882383048534
# 99 300 ====> 0.009319698438048363
# 99 400 ====> 0.011953579261898994