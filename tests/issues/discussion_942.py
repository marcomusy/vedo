from vedo import *


def scroll_left(obj, ename):
    global index
    i = (index - 1) % len(meshes)
    txt.text(meshes[i].name).c("k")
    plt.remove(meshes[index]).add(meshes[i])
    plt.reset_camera()
    index = i

def scroll_right(obj, ename):
    global index
    i = (index + 1) % len(meshes)
    txt.text(meshes[i].name).c("k")
    plt.remove(meshes[index]).add(meshes[i])
    plt.reset_camera()
    index = i

def flag(obj, ename):
    global index
    txt.text("Flag Button Pressed!").c("r")
    plt.reset_camera()


# load some meshes
m1 = Mesh(dataurl + "bunny.obj").c("green5")
m2 = Mesh(dataurl + "apple.ply").c("red5")
m3 = Mesh(dataurl + "beethoven.ply").c("blue5")
m1.name = "a bunny"
m2.name = "an apple"
m3.name = "mr. beethoven"

meshes = [m1, m2, m3]
txt = Text2D(meshes[0].name, font="Courier", pos="top-center", s=1.5)

plt = Plotter()

bu = plt.add_button(
    scroll_right,
    pos=(0.8, 0.06),  # x,y fraction from bottom left corner
    states=[">"],     # text for each state
    c=["w"],          # font color for each state
    bc=["k5"],        # background color for each state
    size=40,          # font size
)
bu = plt.add_button(
    scroll_left,
    pos=(0.2, 0.06),  # x,y fraction from bottom left corner
    states=["<"],     # text for each state
    c=["w"],          # font color for each state
    bc=["k5"],        # background color for each state
    size=40,          # font size
)
bu = plt.add_button(
    flag,
    pos=(0.5, 0.06),
    states=["Flag"],
    c=["w"],
    bc=["r"],
    size=40,
)

index = 0  # init global index
plt += txt
plt.show(meshes[0]).close()