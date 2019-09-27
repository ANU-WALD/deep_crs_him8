from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from matplotlib import pyplot as plt
import numpy as np


def GetModel(conv1_size):
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(400, 400, 8)))
    model.add(Conv2D(32, (conv1_size, conv1_size), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    model.add(Conv2D(1, (1, 1), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

    print(model.summary())

    return model

# Load measured precipitation to create Y (output of the network)
y = np.log(1+np.load("/data/sat_precip/gpm_30.npy")[:,:,:,None])

# Load and normalise satellite reflectances (Try just 3 bands [8,10,14]
b8 = np.load("/data/sat_precip/b8_30.npy")
b9 = np.load("/data/sat_precip/b9_30.npy")
b10 = np.load("/data/sat_precip/b10_30.npy")
b11 = np.load("/data/sat_precip/b11_30.npy")
b12 = np.load("/data/sat_precip/b12_30.npy")
b13 = np.load("/data/sat_precip/b13_30.npy")
b14 = np.load("/data/sat_precip/b14_30.npy")
b15 = np.load("/data/sat_precip/b15_30.npy")

x = np.stack((b8, b9, b10, b11, b12, b13, b14, b15), axis=3)
print(x.shape, y.shape)

# Iterate through different convolution sizes for 1st
for i in [1,3,5]:
    # Instantiate model defined in function above
    model = GetModel(i)

    # Fit data using a 70/30 validation split
    csv_logger = CSVLogger('log_{}.csv'.format(i), append=True, separator=';')
    model.fit(x, y, epochs=50, verbose=1, validation_split=.3, shuffle=True, callbacks=[csv_logger])

    # Save the model once trained for later use
    model.save('cnn{}_rain8.h5'.format(i))
