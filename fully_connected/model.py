from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import multi_gpu_model
import numpy as np
import sys

NUM_GPU = 4

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def GetModel():
    model = Sequential()
    model.add(Dense(2048, activation='relu', input_shape=(1,)))
    model.add(Dropout(.2))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(1, activation='relu'))

    parallel_model = multi_gpu_model(model, NUM_GPU)
    parallel_model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['mae'])
    print(parallel_model.summary())

    return parallel_model

b7 = np.load("/scratch/director2107/CRS_Data/b7.npy")[:200].flatten().reshape(-1,1)
b7 = (b7 - b7.min()) / (b7.max() - b7.min())
b8 = np.load("/scratch/director2107/CRS_Data/b8.npy")[:200].flatten().reshape(-1,1)
b8 = (b8 - b8.min()) / (b8.max() - b8.min())
b9 = np.load("/scratch/director2107/CRS_Data/b9.npy")[:200].flatten().reshape(-1,1)
b9 = (b9 - b9.min()) / (b9.max() - b9.min())
b10 = np.load("/scratch/director2107/CRS_Data/b10.npy")[:200].flatten().reshape(-1,1)
b10 = (b10 - b10.min()) / (b10.max() - b10.min())
b11 = np.load("/scratch/director2107/CRS_Data/b11.npy")[:200].flatten().reshape(-1,1)
b11 = (b11 - b11.min()) / (b11.max() - b11.min())
b12 = np.load("/scratch/director2107/CRS_Data/b12.npy")[:200].flatten().reshape(-1,1)
b12 = (b12 - b12.min()) / (b12.max() - b12.min())
b13 = np.load("/scratch/director2107/CRS_Data/b13.npy")[:200].flatten().reshape(-1,1)
b13 = (b13 - b13.min()) / (b13.max() - b13.min())
b14 = np.load("/scratch/director2107/CRS_Data/b14.npy")[:200].flatten().reshape(-1,1)
b14 = (b14 - b14.min()) / (b14.max() - b14.min())
b15 = np.load("/scratch/director2107/CRS_Data/b15.npy")[:200].flatten().reshape(-1,1)
b15 = (b15 - b15.min()) / (b15.max() - b15.min())
b16 = np.load("/scratch/director2107/CRS_Data/b16.npy")[:200].flatten().reshape(-1,1)
b16 = (b16 - b16.min()) / (b16.max() - b16.min())

bands = [b7, b8, b9, b10, b11, b12, b13, b14, b15, b16]


y = np.load("/scratch/director2107/CRS_Data/crs.npy")[:200].flatten().reshape(-1,1)


i = 7
for b in bands:
    print("Band:", i)
    i += 1
    #x = np.concatenate((b7, b8, b9, b10, b11, b12, b13, b14, b15, b16), axis=1)
    x = b
    print(x.shape)

    #print("X")
    #print(x.mean(), x.max(), x.min())
    #print("Y")
    #print(y.mean(), y.max(), y.min())

    model = GetModel()
    history = model.fit(x, y, epochs=3, verbose=1, batch_size=10000, validation_split=.2, shuffle=True)
