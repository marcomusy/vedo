"""Add an observer to specific objects in a scene"""
from vedo import *

# -----------------------
def func(obj, name=None):
    printc("Plotter callback", c="m")

# -----------------------
def ftxt(obj, ename):
    printc("Text2D callback", obj.__class__.__name__, ename, c="y")
    obj.color(np.random.rand() * 10)

# -----------------------
def fmsh(obj, ename):
    printc("Mesh callback", obj.__class__.__name__, ename, c="b")
    msh.color(np.random.rand() * 10)


msh = Mesh(dataurl + "spider.ply")
cid2 = msh.add_observer("pick", fmsh)

txt = Text2D("CLICK ME", pos="bottom-center", s=3, bg="yellow5").pickable()
cid1 = txt.add_observer("pick", ftxt)

plt = Plotter()
# plt.add_observer("mouse click", func)  ### SAME AS:
# plt.add_callback("mouse click", func, enable_picking=False)
plt.show(txt, msh, __doc__).close()