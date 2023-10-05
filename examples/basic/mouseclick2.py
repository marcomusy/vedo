"""Add observers to specific objects in a scene"""
from vedo import *

def func(obj, name=None):
    printc("Plotter callback")

def ftxt(obj, name):
    print("Text2D callback", type(obj), name)
    obj.color('red')

def fmsh(obj, name):
    print("Mesh callback", type(obj), name)
    msh.color(np.random.rand()*10)

msh = Mesh(dataurl + 'spider.ply')
txt = Text2D("CLICK ME", pos="bottom-center", s=3, bg='yellow5')
txt.pickable()

cid1 = txt.add_observer('pick', ftxt)
cid2 = msh.add_observer('pick', fmsh)

plt = Plotter()
plt.add_observer("mouse click", func)  # same as:
# plt.add_callback("mouse click", func, enable_picking=False)
plt.show(txt, msh, __doc__).close()

