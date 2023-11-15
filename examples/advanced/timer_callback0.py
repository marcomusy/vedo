from vedo import *
from time import time

def loop_func(event):
    msh.rotate_z(0.1)
    txt.text(f"time: {event.time - t0} sec")
    plt.render()

t0 = time()
msh = Cube()
txt = Text2D(bg='yellow', font="Calco")

plt = Plotter(axes=1)
# plt.initialize_interactor() # on windows this is needed
plt.add_callback("timer", loop_func)
plt.timer_callback("start")
plt.show(msh, txt)
plt.close()
