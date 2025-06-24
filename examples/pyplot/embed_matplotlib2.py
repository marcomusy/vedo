import numpy as np
from vedo import Plotter, Points, Image, settings
from vedo.pyplot import histogram, plot
import matplotlib.pyplot as plt


def func(w, evt=""):
    d = data[data > w.value]
    if len(d) == 0:
        return

    ############################################# Vedo Plot
    xy = plot(
        d,
        1 + np.sin(d),
        c="purple4",
        lw=0,
        alpha=0.75,
        aspect=16 / 9,
        title="Vedo Plot of 1+sin(z) vs z",
        xtitle="independent variable",
        ytitle="dependent variable",
    )
    xy = xy.clone2d("top-right", size=0.7, ontop=False)
    xy.name = "myplots"

    ############################################# Vedo Histogram
    hi = histogram(
        d,
        c="orange4",
        alpha=1,
        aspect=16 / 9,
        title="Vedo Histogram",
        xtitle="stochastic variable",
        ytitle="frequency",
        label="my histogram",
        mc="red5",
    )
    hi.add_legend(s=1.2, alpha=0.1)
    hi = hi.clone2d("bottom-right", size=0.7, ontop=True)
    hi.name = "myplots"

    ############################################# Matplotlib histogram
    fig.clf()  # clear the figure
    plt.hist(1-d*d, bins=20, color="green", edgecolor="black")
    img = Image(fig).clone2d("middle-left", size=0.4)
    img.name = "myplots"
    vplt.remove("myplots").add(xy, hi, img)


#############################################################
settings.default_font = "Roboto"

fig = plt.figure(figsize=(8, 6))  # create a matplotlib figure

msh = Points(np.random.randn(1000, 3)).ps(5).c("blue5")
data = msh.points[:, 2]
vplt = Plotter(bg="w", bg2="green9", size=(900, 900))
slider = vplt.add_slider(
    func,
    xmin=-1,
    xmax=1,
    value=0,
    title="slider",
    pos=1,
    # delayed=True,  # update only when the slider is released
)
vplt.show(msh)
vplt.close()
