from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, Dropout
from tensorflow.keras.optimizers import Adam
from matplotlib import pyplot as plt
import numpy as np

def GetModel():
    model = Sequential()
    model.add(BatchNormalization(axis=3, input_shape=(400, 400, 3)))
    # Size 400x400x3
    model.add(Conv2D(32, (5, 5), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    # Size 200x200x32
    model.add(Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    # Size 100x100x64
    model.add(Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    # Size 50x50x128
    model.add(Conv2D(256, (5, 5), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    # Size 25x25x256
    model.add(Conv2DTranspose(128, (5, 5), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    # Size 50x50x128
    model.add(Conv2DTranspose(64, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    # Size 100x100x64
    model.add(Conv2DTranspose(32, (3, 3), strides=(2, 2), activation='relu', padding='same'))
    model.add(BatchNormalization(axis=3))
    # Size 200x200x32
    model.add(Conv2DTranspose(1, (5, 5), strides=(2, 2), activation='relu', padding='same'))
    # Size 400x400x1

    model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.0001), metrics=['mae'])

    print(model.summary())

    return model

# Load and normalise satellite reflectances (Try just 3 bands [8,10,14]
b8 = np.load("/data/sat_precip/b8_30.npy")
b8 = (b8 - b8.mean()) / b8.std()
b10 = np.load("/data/sat_precip/b10_30.npy")
b10 = (b10 - b10.mean()) / b10.std()
b14 = np.load("/data/sat_precip/b14_30.npy")
b14 = (b14 - b14.mean()) / b14.std()

# Stack reflectances in depth to create X (input of the network)
x = np.stack((b8, b10, b14), axis=3)

# Load measured precipitation to create Y (output of the network)
y = np.load("/data/sat_precip/gpm_30.npy")[:,:,:,None]

# Verify dimensions of the data. 
# We should have 1163 samples of 400x400 images for both X and Y
print(x.shape, y.shape)

# Instantiate model defined in function above
model = GetModel()

# Fit data using a 70/30 validation split
history = model.fit(x, y, epochs=10, verbose=1, validation_split=.3, shuffle=True)

# Save the model once trained for later use
#model.save('cnn_rain.h5')

# Generate some data to see if model makes sense
y_pred = model.predict(x[:1, :, :, :])
print(y_pred)
plt.imsave("pred_rain.png", y_pred[0,:,:,0])
plt.imsave("obs_rain.png", y[0,:,:,0])
