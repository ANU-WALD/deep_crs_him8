from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.utils import multi_gpu_model
import numpy as np
import sys

def root_mean_squared_error(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1)) 

def GetModel():
    model = Sequential()
    model.add(Dense(256, activation='relu', input_shape=(6,)))
    model.add(Dropout(.2))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(1, activation='relu'))

    #parallel_model = multi_gpu_model(model, NUM_GPU)
    parallel_model = model
    parallel_model.compile(loss='mse', optimizer=Adam(lr=0.0001), metrics=['mae'])
    print(parallel_model.summary())

    return parallel_model

#b7 = np.load("/scratch/director2107/CRS_Data/b7.npy")[:200].flatten().reshape(-1,1)
#b7 = (b7 - b7.min()) / (b7.max() - b7.min())
wv = np.load("/scratch/director2107/CRS_Data/b8.npy")[:200].flatten().reshape(-1,1)
wv = (wv - wv.min()) / (wv.max() - wv.min())
wv2 = np.square(wv)
wv2 = (wv2 - wv2.min()) / (wv2.max() - wv2.min())
#b8 = np.log(b8)
"""
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
"""
ir = np.load("/scratch/director2107/CRS_Data/b14.npy")[:200].flatten().reshape(-1,1)
ir = (ir - ir.min()) / (ir.max() - ir.min())
ir2 = np.square(ir)
ir2 = (ir2 - ir2.min()) / (ir2.max() - ir2.min())
eir = np.exp(ir)
eir = (eir - eir.min()) / (eir.max() - eir.min())

mul = ir * wv

"""
b15 = np.load("/scratch/director2107/CRS_Data/b15.npy")[:200].flatten().reshape(-1,1)
#b15 = np.log(b15)
b15 = (b15 - b15.min()) / (b15.max() - b15.min())
b16 = np.load("/scratch/director2107/CRS_Data/b16.npy")[:200].flatten().reshape(-1,1)
b16 = (b16 - b16.min()) / (b16.max() - b16.min())
#b10 = np.load("/scratch/director2107/CRS_Data/b10.npy")[:10].flatten().reshape(-1,1)
"""
"""
print(b10.shape)
print(b10[:5])
b12 = np.load("/scratch/director2107/CRS_Data/b12.npy").flatten()
print(b12.shape)
print(b12[:5])
"""
#print(b7.shape, b10.shape)
#x = np.concatenate((b7, b8, b9, b10, b11, b12, b13, b14, b15, b16), axis=1)
#x = np.concatenate((b7, b8, b9, b10, b11, b12, b13, b14, b15, b16), axis=1)
x = np.concatenate((wv, wv2, ir, ir2, eir, mul), axis=1)
print(x.shape)
"""
print(x.shape)
print(x[:5, :])
x = b7
"""
y = np.load("/scratch/director2107/CRS_Data/crs.npy")[:200].flatten().reshape(-1,1)
y = np.log(1+y)
x_train = x[:25000000,:]
y_train = y[:25000000,:]
x_test = x[25000000:,:]
y_test = y[25000000:,:]

print("P (train):")
print("Max:", y_train.max(), "Min:", y_train.min(), y_train.shape)
print("Mean:", y_train.mean(), "MSE(P'=0):", np.mean(np.square(y_train)), "Var:", np.mean(np.square(y_train - np.mean(y_train))))
print("P (validation):")
print("Max:", y_test.max(), "Min:", y_test.min(), y_test.shape)
print("Mean:", y_test.mean(), "MSE(P'=0):", np.mean(np.square(y_test)), "Var:", np.mean(np.square(y_test - np.mean(y_test))))

"""
print("X")
print(x.mean(), x.max(), x.min())
print("Y")
print(y.mean(), y.max(), y.min())
"""

model = GetModel()
history = model.fit(x_train, y_train, epochs=10, verbose=1, batch_size=10000, validation_data=(x_test, y_test))
#history = model.fit(x, y, epochs=30, verbose=1, batch_size=10000, validation_split=.2, shuffle=True)
