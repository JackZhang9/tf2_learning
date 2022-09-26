import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
# from tensorflow import keras
# import seaborn as sns
# from keras.models import Sequential
# from keras.layers import LSTM,Dense,Dropout
# from keras import optimizers
# ### Create the Stacked LSTM model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM, GRU
import tensorflow as tf
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# 设置全部显示
pd.set_option('display.max_rows',None)
pd.set_option('display.max_columns',None)
pd.set_option('display.max_colwidth',200)
pd.set_option('expand_frame_repr',False)

# 导入数据集，训练集，测试集
train_data=pd.read_csv(r'G:\kaggle_datas\daily_climate_time_series_data\DailyDelhiClimateTrain.csv')
test_data=pd.read_csv(r'G:\kaggle_datas\daily_climate_time_series_data\DailyDelhiClimateTest.csv')
# 查看信息
# print(train_data.head())
# print(test_data.shape)

# 检查缺失值,遍历每一列,使用dataframe的.columns属性提取出每一列的列名，然后遍历这个index类型的由列名组成的list对象，如Index(['date','meantemp','humidity'])
print(train_data.columns,type(train_data.columns))
for column in train_data.columns:
    # 用isna()查看每一列有哪些缺失值，并用sum函数统计它们的总数
    print(column,'缺失值数量为：',train_data[column].isna().sum())

# EDA
# 把date数据去除，然后获取列名，训练查看每一列数据
for column in train_data.drop(columns='date').columns:
    # 初始化一块画布，设置尺寸40,8
    plt.figure(figsize=(40, 8))
    # 用列名提取出一列数据，设置线型c.--，
    plt.plot(train_data[column],'c.--', label=column)
    # 设置标题，名为列名
    plt.title(column)
    # # 保存一张图片，名字为列名
    # plt.savefig('lstm-'+column)
    # # 查看图片
    # plt.show()

# preprocessing
# 抽取出温度数据，作为训练标签数据;抽取出湿度，风速，气压数据作为特征数据,抽取出多个数据需要放到一个list中，
x_train = train_data[['humidity', 'wind_speed', 'meanpressure']]
y_train = train_data['meantemp']

# 温度数据处理
# 提取出温度dataframe
temp_df = train_data['meantemp']
# 对温度数据进行归一化操作，通过最小最大缩放MinMaxScaler使数据变化范围变小,构建一个温度缩放器
temp_scaler = MinMaxScaler(feature_range=(0,1))
# 缩放,使用numpy将温度的dataframe转化为numpy ndarray,
temp_df = temp_scaler.fit_transform(np.array(temp_df).reshape(-1,1))

# train dataset and test dataset split
# 取70%的训练集数据做训练集，共1023个数据，30%做测试数据,439个数据
train_dataset_size=int(len(train_data)*0.65)
test_dataset_size=len(train_data)-train_dataset_size
train_dataset=temp_df[0:train_dataset_size]
test_dataset=temp_df[train_dataset_size:len(temp_df)]

# convert an ndarray into a dataframe,build time series data ,x is observe value,y is forecast value
def creat_dataset(dataset,time_step):
    dataX,dataY=[],[]
    # 最后那time_step个是预测不了的，所以长度上减去
    for i in range(len(dataset)-time_step-1):
        # 把一个观察窗口的数据存储为一个list,然后把该观察窗口后的一个值作为预测值
        dataX.append(dataset[i:(i+time_step),0])
        dataY.append(dataset[i+time_step,0])
    # 返回成np.array格式的数据
    return np.array(dataX),np.array(dataY)

# rehsape ndarray into eg:X=t,t+1,t+2,t+3,Y=t+4
# 时间窗口滑动，将时间序列分为前面的观察窗口和后面的预测窗口,设置时间观察窗口为100个时间序列长度，即用每100个数据去预测一个数据
time_step=100
# 训练集的特征和标签，测试集的特征和标签，通过上面的creat_dataset函数，输入训练集和测试集和观察窗口长度，将其转化为训练数据x和预测数据y，可以理解为训练数据x和标签y，数据格式为np.array
x_train_data,y_train_data=creat_dataset(train_dataset,time_step)
x_test_data,y_test_data=creat_dataset(test_dataset,time_step)
# 查看shape
# print(x_train_data,y_train_data.shape)   # (849, 100) (849,),此时是一个二维的矩阵
# print(x_test_data.shape,y_test_data.shape)     # (411, 100) (411,)

# reshape input to be [samples,time_step,feature],which is required for lstm
# 将模型变为适合lstm的格式,reshape()中，第一个值是数据个数，即有多少个数据；第二个值是一个数据有多少个特征，此时就是观察时间窗口的长度；第三个值是1，表示张量个数，此时是1个三维的张量，由多个二维的矩阵构成；此时数据reshape后还是ndarray格式；本来是一张大表，矩阵，现在的shape为(849, 100, 1)，此时是一个三维的张量
x_train_data = x_train_data.reshape(x_train_data.shape[0],x_train_data.shape[1],1)
x_test_data = x_test_data.reshape(x_test_data.shape[0],x_test_data.shape[1],1)
# print('reshape later===>',type(x_train_data),x_train_data.shape,x_train_data)

# model fit
# 模型搭建，搭建lstm模型
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(100,1)))
model.add(GRU(50, return_sequences=True))
model.add(GRU(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
# 查看模型构造
# print(model.summary())

# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm (LSTM)                  (None, 100, 50)           10400
# _________________________________________________________________
# lstm_1 (LSTM)                (None, 100, 50)           20200
# _________________________________________________________________
# lstm_2 (LSTM)                (None, 50)                20200
# _________________________________________________________________
# dense (Dense)                (None, 1)                 51
# =================================================================
# Total params: 50,851
# Trainable params: 50,851
# Non-trainable params: 0
# _________________________________________________________________
# None

# model fitting
# 模型训练,batchsize 64,epoch 120拟合效果最好
model_history = model.fit(x_train_data,y_train_data,validation_data=(x_test_data,y_test_data),batch_size=64,epochs=120,verbose=1)

# model saving


# forecast
# lstm模型做预测，输入变成特定形状的测试数据
train_predict=model.predict(x_train_data)
test_predict=model.predict(x_test_data)

# convert into original form,用缩放，对现在的数据还原成原始数据
train_predict = temp_scaler.inverse_transform(train_predict)
test_predict = temp_scaler.inverse_transform(test_predict)
print('926===>',test_predict)
print('train_predict===>',train_predict[0:10])  # 预测出来的第一个值是第100个值取值，用0-99的值去预测，
print(train_data[100:900]['meantemp'][0:10])
# lstm预测效果不错

# model evalutate
# 模型评估
evaluation = mean_absolute_percentage_error(y_train_data,train_predict)

print(evaluation)


# draw figure
# 训练集和训练集的预测数据
plt.figure(figsize=(30,5))
plt.grid()
plt.plot(y_train_data,label='Train_data')
plt.plot(train_predict,label='Train_predict_data',c='orange')
plt.legend()
plt.title('train and predictions')
# plt.savefig('lstm-train and predictions')
plt.show()

# 损失函数变化,loss
plt.figure(figsize=(30,5))
plt.plot(model_history.history['loss'],label='loss')
plt.plot(model_history.history['val_loss'],label='validation_loss')
plt.legend()
plt.title('loss curve')
# plt.savefig('lstm-loss')
plt.show()


# draw plot
plt.figure(figsize=(30,5))
look_back=100
train_Predict_Plot = np.empty_like(temp_df)
train_Predict_Plot[:, :] = np.nan
train_Predict_Plot[look_back:len(train_predict)+look_back, :] = train_predict
# shift test predictions for plotting
testPredictPlot = np.empty_like(temp_df)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(temp_df)-1, :] = test_predict
# plot baseline and predictions
plt.plot(temp_scaler.inverse_transform(temp_df),label='Temp_scaler')
plt.plot(train_Predict_Plot,label='Train_prediction',c='red')
plt.plot(testPredictPlot,label='Test_prediction',c='orange')
plt.legend()
plt.title('baseline and predictions')
# plt.savefig('lstm-baseline and predictions')
plt.show()




























