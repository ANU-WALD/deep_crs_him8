import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.optimizers import Adam
import numpy as np
import sys

def GetModel():
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(400, 400, 10)))
    model.add(Conv2D(16, (1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(64, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(Dropout(rate=0.2))
    model.add(Conv2D(16, (1, 1), activation='relu', padding='same'))
    model.add(Conv2D(1, (1, 1), activation='relu', padding='same'))

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
x = np.stack((b7, b8, b9, b10, b11, b12, b13, b14, b15, b16), axis=3)
y = np.expand_dims(np.load("/scratch/director2107/CRS_Data/crs.npy"), axis=3)

#x = np.nan_to_num(np.expand_dims(np.load("/scratch/director2107/CRS_Data/b7.npy"), axis=3))
#y = np.nan_to_num(np.expand_dims(np.load("/scratch/director2107/CRS_Data/crs.npy"), axis=3))

print("in mean", x.mean())
print("out mean", y.mean())
print(x.shape)
print(y.shape)

x_train = x[:2000, :, :, :]
y_train = y[:2000, :]

x_test = x[2000:, :, :, :]
y_test = y[2000:, :]
print("train x mean", x[:2000, :].mean())
print("test x mean", x[2000:, :].mean())

model = GetModel()
history = model.fit(x_train, y_train, epochs=30, verbose=1, validation_data=(x_test, y_test))
