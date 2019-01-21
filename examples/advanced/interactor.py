# How to keep interacting with the 3D scene while the program is running
# (by rotating 3D scene with mouse and remain responsive to key press).
# While using the mouse, calls to myloop() function will be suspended.
#
from vtkplotter import Plotter, printc, sin, cube, sphere
from random import uniform as u

vp = Plotter(axes=4)

actor = cube(length=0.2)

counts = 0

def myloop(*event):
   global counts # this is needed because counts is being modified:
   counts += 0.1
 
   # reference 'actor' is not modified so doesnt need to be global
   # (internal properties of the object are, but python doesn't know..)
   actor.pos([0, counts/20, 0]).color(counts) # move cube and change color
 
   rp = [u(-1,1), u(0,1), u(-1,1)] # a random position
   s = sphere(rp, r=0.05, c='blue 0.2')
   vp.render(s, resetcam=True) # show() would cause exiting the loop
 
   # waste cpu time to slow down program
   for i in range(100000): sin(i)
   printc('#', end='')

printc('Interact with the scene, press q to stop execution:', c=1)
# start interaction by rendering actors, call myloop() every millisecond
vp.show(actor, execute=myloop) 
printc('\n bye.')
