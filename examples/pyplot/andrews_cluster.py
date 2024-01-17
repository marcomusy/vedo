"""Andrews curves for the Iris dataset."""
import numpy as np
from sklearn import datasets
from vedo import *
from vedo.pyplot import Figure, plot

iris = datasets.load_iris()  # loading iris data set

AC = andrews_curves(iris.data)
theta = np.linspace(-np.pi, np.pi, 100)

settings.remember_last_figure_format = True
fig = Figure(
    xlim=(-np.pi, np.pi),
    ylim=(-2, 16),
    padding=0,
    axes=dict(htitle="", axes_linewidth=2, xyframe_line=0),
)

setosas = []
for r in AC[:20]:  # setosa
    p = pol2cart(r, theta).T
    fig += plot(theta, r, c="red5")
    setosas.append(Line(p))
setosas = merge(setosas).lw(3).c("red5")

versicolors = []
for r in AC[50:70]:  # versicolor
    p = pol2cart(r, theta).T
    fig += plot(theta, r, c="blue5")
    versicolors.append(Line(p))
versicolors = merge(versicolors).lw(3).c("blue5")

virginicas = []
for r in AC[100:120]:  # virginica
    p = pol2cart(r, theta).T
    fig += plot(theta, r, c="green5")
    virginicas.append(Line(p))
virginicas = merge(virginicas).lw(3).c("green5")

fig = fig.clone2d(size=0.75)
show(setosas, versicolors, virginicas, fig, __doc__, axes=12)
