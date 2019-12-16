from keras.datasets import boston_housing
from keras.datasets import reuters
import numpy as np
import os
import tensorflow as tf
from keras import models
from keras import layers
from keras import optimizers
from keras import losses
from keras import metrics
import matplotlib.pyplot as plt
import time
(train_data, train_targets),(test_data, test_targets)=boston_housing.load_data()
mean = train_data.mean(axis=0)
train_data -=mean
std = train_data.std(axis=0)
train_data/=std
test_data-=mean
test_data/=std
def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                          input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))#非0~1的預測,故不使用啟動函數轉為01
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model
              
k=4
num_val_samples = len(train_data)//k
num_epochs = 200
all_scores=[]

tStart = time.time()
all_mae_histories=[]
for i in range(k):
    print('processing fold #',i)
    val_data = train_data[i * num_val_samples: (i+1) * num_val_samples ]
    val_targets = train_targets[i * num_val_samples:(i+1) * num_val_samples]
    partial_train_data = np.concatenate([train_data[:i * num_val_samples ],
                                        train_data[(i+1)*num_val_samples:]],axis=0)
    partial_train_targets = np.concatenate([train_targets[:i * num_val_samples ],
                                        train_targets[(i+1)*num_val_samples:]],axis=0)
    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets,validation_data=(val_data, val_targets),epochs = num_epochs,batch_size=1,verbose=0)
    mae_history = history.history['val_mae']
    all_mae_histories.append(mae_history)
tEnd = time.time()
print ("It cost %f sec" % (tEnd - tStart))#會自動做近位
print (tEnd - tStart)#原型長這樣