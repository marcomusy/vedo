"""Create three checkbox buttons to toggle objects on/off."""
from vedo import Mesh, Plotter, dataurl

s1 = Mesh(dataurl+"bunny.obj").normalize().x(0).color("p5")
s2 = Mesh(dataurl+"teapot.vtk").normalize().x(3).rotate_x(-90).color("y5")
s3 = Mesh(dataurl+"mug.ply").normalize().x(6).color("r5")


def func1(b, evt):
    s1.toggle()  # toggle visibility
    b.switch()

def func2(b, evt):
    s2.toggle()
    b.switch()

def func3(b, evt):
    s3.toggle()
    b.switch()

def func4(_, evt):
    [s.toggle() for s in (s1,s2,s3)]
    [b.switch() for b in plt.buttons]

plt = Plotter(axes=1, size=(1000,500))
plt.add_hint(s1, "A Bunny",  size=42) # shows a label when hovering on the object
plt.add_hint(s2, "A Teapot", size=42)
plt.add_hint(s3, "A Mug",    size=42)

plt.add_button(func1, pos=(0.4,0.15), size=42, states=["", ""], bc=["p5", "k7"])
plt.add_button(func2, pos=(0.5,0.15), size=42, states=["", ""], bc=["y5", "k7"])
plt.add_button(func3, pos=(0.6,0.15), size=42, states=["", ""], bc=["r5", "k7"])
plt.add_button(func4, pos=(0.9,0.15), size=42, states=["flip","flip"], bc=["k4","k5"],
               c=["k5","k4"], font="Cartoons123")

plt.show(s1, s2, s3, __doc__, zoom=1.8).close()
