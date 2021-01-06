from vedo import *

################################################################
def onLeftClick(mesh):
    if not mesh: return
    cpt = vector(mesh.picked3d)+[0,0,2]
    printc("Added point:", precision(cpt[:2],4), c='g')
    cpoints.append(cpt)
    update()

def onRightClick(mesh):
    if not mesh or len(cpoints)==0: return
    p = cpoints.pop() # pop removes from the list the last obj
    plt.actors.pop()
    printc("Deleted point:", precision(p[:2], 4), c="r")
    update()

def update():
    global spline, points
    plt.remove([spline, points], render=False)
    points = Points(cpoints, r=8).c('violet').alpha(0.8)
    spline = None
    if len(cpoints)>2:
        spline = Spline(cpoints, smooth=0).c('yellow').alpha(0.8)
    plt.add([spline, points])

def keyfunc(key):
    global spline, points, cpoints
    if key == 'c':
        plt.remove([spline, points], render=True)
        cpoints = []
        points = None
        spline = None
        printc("==== Cleared all points ====", c="r")
    elif key == 's':
        with open(outfl, 'w') as f:
            # uncomment the second line to save the spline instead (with 100 pts)
            f.write(str(vector(cpoints)[:,(0,1)])+'\n')
            #f.write(str(Spline(cpoints, smooth=0, res=100).points()[:,(0,1)])+'\n')
            printc("\nCoordinates saved to file:", outfl, c='y', invert=1)
    else:
        printc('key press:', key, 'ignored')


############################################################
outfl = 'spline.txt'
cpoints = []
points, spline= None, None

plt = Plotter()
plt.keyPressFunction = keyfunc  # make keyfunc known to Plotter class
plt.mouseLeftClickFunction = onLeftClick
plt.mouseRightClickFunction= onRightClick

pic = plt.load("https://embryology.med.unsw.edu.au/embryology/images/4/40/Mouse-_embryo_E11.5.jpg")
pic.alpha(0.99).pickable(True)

t = """Click to add a point
Right-click to remove
Press c to clear points
Press s to save to file"""
instr = Text2D(t, pos='bottom-left', c='white', bg='green', font='Quikhand', s=0.9)

# make a transparent box around the image
plt.show(pic, instr, axes=True, bg='blackboard')


