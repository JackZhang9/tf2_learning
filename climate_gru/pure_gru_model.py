import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential,save_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from sklearn.preprocessing import MinMaxScaler
from tensorflow import saved_model
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

# model fit
model = Sequential()
model.add(GRU(50, return_sequences=True, input_shape=(100,1)))
model.add(GRU(50, return_sequences=True))
model.add(GRU(50))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# model fitting
model_history = model.fit(x_train_data,y_train_data,validation_data=(x_test_data,y_test_data),batch_size=64,epochs=120,verbose=1)

# model saving
model.save('./keras_model/1/',save_format='tf')






























