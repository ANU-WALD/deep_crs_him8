from sklearn.linear_model import LinearRegression
import numpy as np
import sys

print("AAA1")
b7 = np.load("/scratch/director2107/CRS_Data/b7.npy")
#b8 = np.load("/scratch/director2107/CRS_Data/b8.npy")
#b9 = np.load("/scratch/director2107/CRS_Data/b9.npy")
b10 = np.load("/scratch/director2107/CRS_Data/b10.npy")
#b11 = np.load("/scratch/director2107/CRS_Data/b11.npy")
b12 = np.load("/scratch/director2107/CRS_Data/b12.npy")
#b13 = np.load("/scratch/director2107/CRS_Data/b13.npy")
#b14 = np.load("/scratch/director2107/CRS_Data/b14.npy")
#b15 = np.load("/scratch/director2107/CRS_Data/b15.npy")
#b16 = np.load("/scratch/director2107/CRS_Data/b16.npy")
#x = np.stack((b7, b8, b9, b10, b11, b12, b13, b14, b15, b16), axis=3).reshape(-1, 10)
x = np.stack((b7, b10, b12), axis=3).reshape(-1, 3)
#x = b7.flatten()
y = np.load("/scratch/director2107/CRS_Data/crs.npy").flatten()
print("AAA2")

reg = LinearRegression().fit(x, y)
print(reg.score(x, y))
