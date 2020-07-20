"""Draw a spline on a 2D image"""
from vedo import Plotter, Points, KSpline, Text2D, datadir

##############################################################################
cpoints, points, spline = [], None, None

def onLeftClick(mesh):
    if not mesh: return
    cpoints.append(mesh.picked3d)
    print('point:', mesh.picked3d[:2])
    update()

def onRightClick(mesh):
    if len(cpoints)==0: return
    cpoints.pop()
    plt.actors.pop()
    update()

def update():
    global spline, points
    plt.resetcam = False
    plt.remove([spline, points], render=False)
    points = Points(cpoints).c('violet')
    spline = None
    if len(cpoints)>1:
        spline = KSpline(cpoints).c('yellow').alpha(0.5)
    plt.add([spline, points])

def keyfunc(key):
    global spline, points, cpoints
    if key == 'c':
        plt.remove([spline, points])
        cpoints, points, spline= [], None, None

##############################################################################
plt = Plotter()
plt.keyPressFunction = keyfunc
plt.mouseLeftClickFunction = onLeftClick
plt.mouseRightClickFunction= onRightClick

t = """Click to add a point
Right-click to remove it
Press c to clear points"""
msg = Text2D(t, pos="bottom-left", c='k', bg='green', font='Quikhand', s=0.9)

pic = plt.load(datadir+"images/dog.jpg")
# make a transparent box around the picture for clicking
box = pic.box(pad=1).wireframe(False).alpha(0.01)

plt.show(__doc__, pic, box, msg, axes=True, interactorStyle=6, bg='w')
