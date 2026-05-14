import zarr
import matplotlib.pyplot as plt
import numpy as np

test = zarr.open('/Volumes/My Passport/Sasaki/MTsingleBeads/beads3um/20260512/MTs_order_parameter.zarr', mode='r')
y = test[:]

plt.plot(np.arange(len(y))*4, y)
plt.show()