import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys

def GetModel():
    model = Sequential()
    model.add(Dense(16, activation='relu', input_shape=(10,)))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='relu'))

    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

    print(model.summary())

    return model

b7 = np.load("/scratch/director2107/CRS_Data/b7.npy")
b8 = np.load("/scratch/director2107/CRS_Data/b8.npy")
b9 = np.load("/scratch/director2107/CRS_Data/b9.npy")
b10 = np.load("/scratch/director2107/CRS_Data/b10.npy")
b11 = np.load("/scratch/director2107/CRS_Data/b11.npy")
b12 = np.load("/scratch/director2107/CRS_Data/b12.npy")
b13 = np.load("/scratch/director2107/CRS_Data/b13.npy")
b14 = np.load("/scratch/director2107/CRS_Data/b14.npy")
b15 = np.load("/scratch/director2107/CRS_Data/b15.npy")
b16 = np.load("/scratch/director2107/CRS_Data/b16.npy")
x = np.stack((b7, b8, b9, b10, b11, b12, b13, b14, b15, b16), axis=3).reshape(-1, 10)
y = np.load("/scratch/director2107/CRS_Data/crs.npy").flatten()

model = GetModel()
history = model.fit(x, y, epochs=30, verbose=1, validation_split=.2, shuffle=True)
