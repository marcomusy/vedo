"""Fourier 2D shape reconstruction with epicycles representation"""
# Original version from D. Shiffman (2019), adapted for vedo by M. Musy (2022)
# https://thecodingtrain.com/CodingChallenges/130.2-fourier-transform-drawing.html
import numpy as np
import vedo

order = 20  # restrict to this nr of fourier coefficients in reconstruction


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
    return vedo.utils.sort_by_column(X, 3, invert=True)


def epicycles(time, rotation, fourier, order):
    global objs
    plt.remove(objs)
    x, y = [0, 0]
    objs, path = [], []
    for i in range(len(fourier[:order])):
        re, im, freq, amp, phase = fourier[i]
        if amp > 0.2:
            c = vedo.Circle([x,y], amp).wireframe().lw(1)
            objs.append(c)
        x += amp * np.cos(freq * time + phase + rotation)
        y += amp * np.sin(freq * time + phase + rotation)
        path.append([x,y])

    if len(points)>0:
        hline = vedo.Line([x,y], points[-1]).c('red5').lw(1)
        pline = vedo.Line(path).c('green5').lw(2)
        oline = vedo.Line(points).c('red4').lw(5)
        objs += [hline, pline, oline]
        plt.add(objs).render()
    return [x, y]


# Load some 2D shape and make it symmetric
shaper  = vedo.Assembly(vedo.dataurl+'timecourse1d.npy')[55]
# shaper = shape.clone().mirror('x').reverse()
shapel = vedo.Line(np.flip(shaper.clone().mirror('x').coordinates, axis=0))
shape = vedo.merge(shaper, shapel)
x, y, _ = shape.vertices.T

# Compute Fourier Discrete Transform in x and y separately:
fourierX = DFT(x)
fourierY = DFT(y)

vedo.settings.default_font = 'Glasgo'

plt = vedo.Plotter(size=(1500,750), bg='black', axes=1, interactive=False)
txt = vedo.Text2D(f"{__doc__} (order={order})", c='red9', bg='white', pos='bottom-center')
plt.show(shape, txt, mode='image', zoom=1.9)

objs, points = [], []
times = np.linspace(0, 2*np.pi, len(fourierX), endpoint=False)
for time in times:
    x, _ = epicycles(time,       0, fourierX, order)
    _, y = epicycles(time, np.pi/2, fourierY, order)
    points.append([x, y])

plt.interactive().close()

