import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from vtkplotter import load, Volume, show, datadir

maxradius = 0.2
neurons = 30
epochs = 20

image = load(datadir+"embryo.tif").imagedata()

vmin, vmax = image.GetScalarRange()
nx, ny, nz = image.GetDimensions()
print("Scalar Range:", vmin, vmax, "Dimensions", image.GetDimensions())

visdata = np.zeros([nx, ny, nz])
datalist, scalars = [], []
lsx = np.linspace(0, 1, nx, endpoint=False)
lsy = np.linspace(0, 1, ny, endpoint=False)
lsz = np.linspace(0, 1, nz, endpoint=False)
for i, x in enumerate(lsx):
    for j, y in enumerate(lsy):
        for k, z in enumerate(lsz):
            s = image.GetScalarComponentAsDouble(i, j, k, 0)
            s = (s - vmin) / (vmax - vmin)
            visdata[i, j, k] = s
            datalist.append([x, y, z])
            scalars.append(s)
datalist = np.array(datalist)
scalars = np.array(scalars)

# random shuffle to make sure order doesnt matter
s = list(range(len(scalars)))
np.random.shuffle(s)
shuffled_datalist = np.array(datalist[s])
shuffled_scalars  = np.array(scalars[s])

model = Sequential()
model.add(Dense(neurons, activation="relu", input_dim=3))
model.add(Dense(neurons, activation="relu"))
model.add(Dense(neurons, activation="relu"))
model.add(Dense(1,       activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

model.fit(shuffled_datalist, shuffled_scalars,
          epochs=epochs, batch_size=max(nx,ny,nz))

predicted_scalars = model.predict(datalist)
model.summary()

idx = 0
vispred = np.zeros([nx, ny, nz])
for i, x in enumerate(lsx):
    for j, y in enumerate(lsy):
        for k, z in enumerate(lsz):
            vispred[i, j, k] = predicted_scalars[idx]
            idx += 1

v1 = Volume(visdata)
v2 = Volume(vispred)
s1 = v1.isosurface(threshold=0).alpha(0.8)
s2 = v2.isosurface(threshold=0).alpha(0.8)

show(v1, v2, s1, s2, N=4, axes=8, bg="w")
