from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import sys

def GetModel():
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(400, 400, 5)))
    model.add(Conv2D(16, (1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, (1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(32, (1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(16, (1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(1, (1, 1), activation='relu', padding='same'))

    model.compile(loss='mean_absolute_error', optimizer=Adam(lr=0.01), metrics=['mae'])

    print(model.summary())

    return model

b7 = np.load("/scratch/director2107/CRS_Data/b7.npy")
b10 = np.load("/scratch/director2107/CRS_Data/b10.npy")
b12 = np.load("/scratch/director2107/CRS_Data/b12.npy")
x = np.stack((b7, b10, b12), axis=3)
y = np.expand_dims(np.load("/scratch/director2107/CRS_Data/crs.npy"), axis=3)

#x = np.nan_to_num(np.expand_dims(np.load("/scratch/director2107/CRS_Data/b7.npy"), axis=3))
#y = np.nan_to_num(np.expand_dims(np.load("/scratch/director2107/CRS_Data/crs.npy"), axis=3))

print("in mean", x.mean())
print("out mean", y.mean())
sys.exit()
print("out mean", np.nanmean(y), np.nanmin(y), np.nanmax(y))
print(x.shape)
print(y.shape)

x_train = x[:80, :, :, :]
y_train = y[:80, :]

x_test = x[80:, :, :, :]
y_test = y[80:, :]
print("train x mean", x[:80, :].mean())
print("test x mean", x[80:, :].mean())

model = GetModel()
history = model.fit(x_train, y_train, epochs=30, verbose=1, validation_data=(x_test, y_test))
