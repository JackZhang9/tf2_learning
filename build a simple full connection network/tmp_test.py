# using tensorflow to build a simple full connection neural network with 2 hidden layers
# 用tensorflow搭建一个简单2层隐藏层的全连接神经网络
import tensorflow as tf

# 1.instance a sequential model
model = tf.keras.Sequential()
# 2.add the first dense layer,you should give a neural node numbers,by the way you could give a input shape
model.add(tf.keras.layers.Dense(8, input_shape=(16,)))
# 3.add the second dense layer,you could not input shape now
model.add(tf.keras.layers.Dense(4))
# 4.check model length
l=len(model.weights)
# 5.check model weights
print(l,model.weights)
# 6.check model summary
print(model.summary())

# it's 172 parameters in this model
# there is 16*8 weights and 8 bias in first layer and the parameter's amount is 136,
# and the second layer has 8*4 weights and 4 bias, and the parameter's amount is 36,
# 模型参数一共172，
# 第一层有16*8个weight和8个bias，共136个参数；第二层有8*4个weight和4个bias，共36个参数；
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# dense (Dense)                (None, 8)                 136
# _________________________________________________________________
# dense_1 (Dense)              (None, 4)                 36
# =================================================================
# Total params: 172
# Trainable params: 172
# Non-trainable params: 0
# _________________________________________________________________
# None