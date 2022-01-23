"""Fourier 2D shape reconstruction with epicycles (order=50)"""
# Original version from D. Shiffman (2019), adapted for vedo by M. Musy (2022)
# https://thecodingtrain.com/CodingChallenges/130.2-fourier-transform-drawing.html
import numpy as np
import vedo


def DFT(x):
    X = []
    N = len(x)
    for freq in range(N):
        re, im = [0, 0]
        for n in range(N):
            phi = (2 * np.pi * freq * n) / N
            re += x[n] * np.cos(phi)
            im -= x[n] * np.sin(phi)
        re, im = [re/N, im/N]
        amp = np.sqrt(re*re + im*im)
        phase = np.arctan2(im, re)
        X.append([re, im, freq, amp, phase])
    return vedo.utils.sortByColumn(X, 3, invert=True)


def epicycles(time, rotation, fourier):
    global objs
    plt.remove(objs)
    objs = []
    x, y = [0, 0]
    path = []
    for i in range(len(fourier[:order])):
        re, im, freq, amp, phase = fourier[i]
        if amp > 0.1:
            c = vedo.Circle([x,y], amp).wireframe().lw(0.1)
            objs.append(c)
        x += amp * np.cos(freq * time + phase + rotation)
        y += amp * np.sin(freq * time + phase + rotation)
        path.append([x,y])

    if len(points):
        hline = vedo.Line([x,y], points[-1], c='green5', lw=0.1)
        pline = vedo.Line(path, c='green5', lw=2)
        oline = vedo.Line(points, c='red4', lw=5)
        objs += [hline, pline, oline]
        plt.add(objs, resetcam=False)

    return [x, y]


# Load some 2D shape
shape = vedo.load('data/timecourse1d.npy')[55]
x, y, _ = shape.points().T

# Compute Fourier Discrete Transform in x and y separately:
order = 50
fourierX = DFT(x)
fourierY = DFT(y)

plt = vedo.Plotter(bg='black', axes=1, interactive=False)
plt.show(shape.z(-0.1), __doc__)

objs, points = [], []
times = np.linspace(0, 2*np.pi, len(fourierX), endpoint=False)
for time in times:
    x, _ = epicycles(time,       0, fourierX)
    _, y = epicycles(time, np.pi/2, fourierY)
    points.append([x, y])

vedo.interactive()

