import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys

def GetModel():
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(400, 400, 3)))
    model.add(Conv2D(16, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(64, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(16, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(1, (1, 1), activation='relu', padding='same'))

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001), metrics=['mae'])

    print(model.summary())

    return model

b7 = np.load("/scratch/director2107/CRS_Data/b7.npy")
b7 = (b7 - b7.min()) / (b7.max() - b7.min())
"""
b8 = np.load("/scratch/director2107/CRS_Data/b8.npy")
b8 = (b8 - b8.min()) / (b8.max() - b8.min())
b9 = np.load("/scratch/director2107/CRS_Data/b9.npy")
b9 = (b9 - b9.min()) / (b9.max() - b9.min())
b10 = np.load("/scratch/director2107/CRS_Data/b10.npy")
b10 = (b10 - b10.min()) / (b10.max() - b10.min())
b11 = np.load("/scratch/director2107/CRS_Data/b11.npy")
b11 = (b11 - b11.min()) / (b11.max() - b11.min())
"""
b12 = np.load("/scratch/director2107/CRS_Data/b12.npy")
b12 = (b12 - b12.min()) / (b12.max() - b12.min())
"""
b13 = np.load("/scratch/director2107/CRS_Data/b13.npy")
b13 = (b13 - b13.min()) / (b13.max() - b13.min())
"""
b14 = np.load("/scratch/director2107/CRS_Data/b14.npy")
b14 = (b14 - b14.min()) / (b14.max() - b14.min())
"""
b15 = np.load("/scratch/director2107/CRS_Data/b15.npy")
b15 = (b15 - b15.min()) / (b15.max() - b15.min())
b16 = np.load("/scratch/director2107/CRS_Data/b16.npy")
b16 = (b16 - b16.min()) / (b16.max() - b16.min())
"""

x = np.stack((b7, b12, b14), axis=3)[:200,:]
#x = np.stack((b7, b8, b9, b10, b11, b12, b13, b14, b15, b16), axis=3)
y = np.expand_dims(np.load("/scratch/director2107/CRS_Data/crs.npy"), axis=3)[:200,:]
#y = (y - y.min()) / (y.max() - y.min())

print("in mean", x.mean())
print("out mean", y.mean())
print(x.shape)
print(y.shape)

model = GetModel()
history = model.fit(x, y, epochs=300, verbose=1, validation_split=.2, shuffle=True)
