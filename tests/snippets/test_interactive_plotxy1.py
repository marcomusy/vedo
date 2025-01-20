"""Interactive plot with a slider
to change the k value of a sigmoid function."""
import numpy as np
from vedo import Plotter, settings
from vedo.pyplot import plot

kinit = 0.75
n = 3
x = np.linspace(-1, 1, 100)

def update_plot(widget=None, event=""):
    k = widget.value if widget else kinit
    # y = 1/(1 + (k/x)**n) # hill function
    # y = np.abs(k*x)**n *np.sign(k*x) # power law
    y = (2 / (1 + np.exp(-np.abs(n*k*x))) - 1) *np.sign(k*x) # sigmoid
    p = plot(x, y, c='red5', lw=4, xlim=(-1,1), ylim=(-1,1), aspect=1)
    plt.remove("PlotXY").add(p)

settings.default_font = "Roboto"
plt = Plotter(size=(900, 1050))
plt.add_slider(update_plot, -1, 1, value=kinit, title="k value", c="red3")
update_plot()
plt.show(__doc__, mode="2d", zoom='tight')
