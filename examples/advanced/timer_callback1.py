"""Create a simple Play/Pause app with a timer event
You can interact with the scene during the loop!
..press q to quit"""
import time
import numpy as np
from vedo import Plotter
from vedo.pyplot import plot


def bfunc(obj, ename):
    global timer_id
    plotter.timer_callback("destroy", timer_id)
    if "Play" in button.status():
        # instruct to call handle_timer() every 10 msec:
        timer_id = plotter.timer_callback("create", dt=10)
    button.switch()

def handle_timer(event):
    t = time.time() - t0
    x = np.linspace(t, t + 4*np.pi, 50)
    y = np.sin(x) * np.sin(x/12)
    fig = plot(
        x, y, '-o', ylim=(-1.2, 1.2), aspect=3/1,
        xtitle="time window [s]", ytitle="intensity [a.u.]",
    )
    fig.shift(-x[0]) # put the whole plot object back at (0,0)
    # Pop (remove) the old plot and add the new one
    plotter.pop().add(fig).render()


timer_id = -1
t0 = time.time()

plotter= Plotter(size=(1200,600))
# plt.initialize_interactor() # on windows this is needed

button = plotter.add_button(bfunc, states=[" Play ","Pause"], size=40)
evntid = plotter.add_callback("timer", handle_timer, enable_picking=False)

x = np.linspace(0, 4*np.pi, 50)
y = np.sin(x) * np.sin(x/12)
fig = plot(x, y, ylim=(-1.2, 1.2), xtitle="time", aspect=3/1, lc='grey5')

plotter.show(__doc__, fig, zoom=2)
