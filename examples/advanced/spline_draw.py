from vedo import *

################################################################
def onLeftClick(evt):
    if not evt.actor: return
    cpt = vector(evt.picked3d) + [0,0,1]
    printc("Added point:", precision(cpt[:2],4), c='g')
    cpoints.append(cpt)
    update()

def onRightClick(evt):
    if not evt.actor or len(cpoints)==0: return
    p = cpoints.pop() # pop removes from the list the last obj
    plt.actors.pop()
    printc("Deleted point:", precision(p[:2], 4), c="r")
    update()

def update():
    global spline, points
    plt.remove([spline, points])
    points = Points(cpoints, r=8).c('violet').alpha(0.8)
    spline = None
    if len(cpoints)>2:
        spline = Spline(cpoints, closed=True).c('yellow').alpha(0.8)
        # spline.ForceOpaqueOn()  # VTK9 has problems with opacity
        # points.ForceOpaqueOn()
    plt.add([points, spline])

def keyfunc(evt):
    global spline, points, cpoints
    if evt.keyPressed == 'c':
        plt.remove([spline, points], render=True)
        cpoints = []
        points = None
        spline = None
        printc("==== Cleared all points ====", c="r")
    elif evt.keyPressed == 's':
        with open(outfl, 'w') as f:
            # uncomment the second line to save the spline instead (with 100 pts)
            f.write(str(vector(cpoints)[:,(0,1)])+'\n')
            #f.write(str(Spline(cpoints, smooth=0, res=100).points()[:,(0,1)])+'\n')
            printc("\nCoordinates saved to file:", outfl, c='y', invert=1)
    else:
        printc('key press:', evt.keyPressed)


############################################################
outfl = 'spline.txt'
cpoints = []
points, spline= None, None

pic = Picture(dataurl+"images/Mouse-_embryo_E11.5.jpg",
              channels=[0,1,2]) # keep rgb but drop alpha channel

t = """Click to add a point
Right-click to remove
Drag mouse to change constrast
Press c to clear points
Press s to save to file"""
instrucs = Text2D(t, pos='bottom-left', c='white', bg='green', font='Quikhand', s=0.9)

plt = Plotter()
plt.addCallback('KeyPress', keyfunc)
plt.addCallback('LeftButtonPress', onLeftClick)
plt.addCallback('RightButtonPress', onRightClick)
plt.show(pic, instrucs, axes=True, bg='blackboard', mode='image').close()


