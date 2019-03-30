import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import xarray as xr


fig = plt.figure()
ds = xr.open_dataset("/data/sat_precip/H8_Flow.nc")
print(ds)
print(ds.time)
#print(ds["B7"][:,:,:].min())
#print(ds["B7"][:,:,:].max())
#sys.exit()

ims = []
for i in range(350):
    im = plt.imshow(ds["B7"][i,:,:], animated=True, vmin=200, vmax=400)
    ims.append([im])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)

ani.save('clouds.mp4')
