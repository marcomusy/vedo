"""Closing the Rendering Window

Press q:
Control returns to terminal,
window will not close but become unresponsive"""
from vtkplotter import Paraboloid, Hyperboloid, Plotter, show

mesh = Paraboloid()

vp1 = show(mesh, __doc__, title='First Plotter instance')

# Now press 'q' to exit the window interaction,
# windows stays open but not reactive anymore.

# You can go back to interavtion mode by simply calling:
#show()

input('\nControl returned to terminal shell:\nwindow is now unresponsive (press Enter)')

vp1.closeWindow()

# window should now close, the Plotter instance becomes unusable
# but mesh objects still exist in it:

print("First Plotter actors:", vp1.actors)
vp1.show()  # THIS HAS NO EFFECT: window does not exist anymore. Cannot reopen.

##################################################################
# Can now create a brand new Plotter and show the old object in it
vp2 = Plotter(title='Second Plotter instance', pos=(500,0))
vp2.show(vp1.actors[0].color('red'))

##################################################################
# Create a third new Plotter and then close the second
vp3 = Plotter(title='Third Plotter instance')

vp2.closeWindow()
print('vp2.closeWindow() called')

vp3.show(Hyperboloid())

from vtkplotter import closeWindow
closeWindow()  # automatically find and close the current window

print('done.')
