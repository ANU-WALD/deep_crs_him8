from tensorflow.keras.models import load_model
import numpy as np
import sys

a = np.load("prec_pred.npy")
print(a.max(), a.min())
sys.exit()

a = np.load("/scratch/director2107/CRS_Data/b8.npy")[:300]#.flatten().reshape(-1,1)
a2 = np.load("b8.npy")#.flatten().reshape(-1,1)
a2 = (a2 - a.min()) / (a.max() - a.min())
print(a.shape)
b = np.load("/scratch/director2107/CRS_Data/b14.npy")[:300]#.flatten().reshape(-1,1)
b2 = np.load("b14.npy")#.flatten().reshape(-1,1)
b2 = (b2 - b.min()) / (b.max() - b.min())
print("Ya")

x = np.concatenate((a2.flatten().reshape(-1,1), b2.flatten().reshape(-1,1)), axis=1)
print(x.shape)

model = load_model('bom_crs.h5')
y_hat = model.predict(x)
np.save("prec_pred", y_hat.reshape((5500,5500)))
