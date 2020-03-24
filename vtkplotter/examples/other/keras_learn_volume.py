import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from vtkplotter import Volume, show, arange, settings

settings.useDepthPeeling = True

n = 10
neurons = 60

visdata = np.zeros([n, n, n])
datalist, scalars = [], []
ls = np.linspace(0, 1, n, endpoint=False)
for i, x in enumerate(ls):
    for j, y in enumerate(ls):
        for k, z in enumerate(ls):
            s = (np.sin(x * 4) ** 2 + np.cos(y * 3) ** 2 + np.sqrt(z)) / 3
            visdata[i, j, k] = s
            datalist.append([x, y, z])
            scalars.append(s)
datalist = np.array(datalist)
scalars = np.array(scalars)

model = Sequential()
model.add(Dense(neurons, activation="relu", input_dim=3))
model.add(Dense(neurons, activation="relu"))
model.add(Dense(neurons, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])

model.fit(datalist, scalars, epochs=50, batch_size=64)

predicted_scalars = model.predict(datalist)

model.summary()

idx = 0
vispred = np.zeros([n, n, n])
for i, x in enumerate(ls):
    for j, y in enumerate(ls):
        for k, z in enumerate(ls):
            vispred[i, j, k] = predicted_scalars[idx]
            idx += 1

v1 = Volume(visdata)
v2 = Volume(vispred)

s1 = v1.isosurface(threshold=[t for t in arange(0, 1, 0.1)])
s1.alpha(0.5)

s2 = v2.isosurface(threshold=[t for t in arange(0, 1, 0.1)])
s2.alpha(0.5)

show([[v1, s1], s2], N=2, axes=8)
