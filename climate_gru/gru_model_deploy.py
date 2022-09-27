import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt

train_data=pd.read_csv(r'G:\kaggle_datas\daily_climate_time_series_data\DailyDelhiClimateTrain.csv')
test_data=pd.read_csv(r'G:\kaggle_datas\daily_climate_time_series_data\DailyDelhiClimateTest.csv')

temp_df = train_data['meantemp']
temp_scaler = MinMaxScaler(feature_range=(0,1))
temp_df = temp_scaler.fit_transform(np.array(temp_df).reshape(-1,1))

train_dataset_size=int(len(train_data)*0.65)
test_dataset_size=len(train_data)-train_dataset_size
train_dataset=temp_df[0:train_dataset_size]
test_dataset=temp_df[train_dataset_size:len(temp_df)]

def creat_dataset(dataset,time_step):
    dataX,dataY=[],[]
    for i in range(len(dataset)-time_step-1):
        dataX.append(dataset[i:(i+time_step),0])
        dataY.append(dataset[i+time_step,0])
    return np.array(dataX),np.array(dataY)

time_step=100
x_train_data,y_train_data=creat_dataset(train_dataset,time_step)
x_test_data,y_test_data=creat_dataset(test_dataset,time_step)

x_train_data = x_train_data.reshape(x_train_data.shape[0],x_train_data.shape[1],1)
x_test_data = x_test_data.reshape(x_test_data.shape[0],x_test_data.shape[1],1)

# model load,reload model
model=tf.keras.models.load_model('./keras_model/1/')
f=model.signatures['serving_default']
print(model)

train_predict=model.predict(x_train_data)
test_predict=model.predict(x_test_data)

# draw figure
# 训练集和训练集的预测数据
plt.figure(figsize=(30,5))
plt.grid()
plt.plot(y_train_data,label='Train_data')
plt.plot(train_predict,label='Train_predict_data',c='orange')
plt.legend()
plt.title('train and predictions')
plt.show()

train_predict = temp_scaler.inverse_transform(train_predict)
test_predict = temp_scaler.inverse_transform(test_predict)
pd.DataFrame(train_predict).to_csv('train_predict.csv')
pd.DataFrame(test_predict).to_csv('test_predict.csv')
print(train_predict)
print(test_predict)

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
plt.show()

print(model.summary())









































