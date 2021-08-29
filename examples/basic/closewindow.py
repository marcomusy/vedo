"""Closing the Rendering Window

Press q:
Control returns to terminal,
window will not close but become unresponsive"""
from vedo import *

mesh = Paraboloid()

vp1 = show(mesh, __doc__, title='First Plotter instance')

# Now press 'q' to exit the window interaction,
# windows stays open but not reactive anymore.

# You can go back to interavtion mode by simply calling:
#show()

printc('\nControl returned to terminal shell:', c='tomato', invert=1)
ask('window is now unresponsive (press Enter here)', c='tomato', invert=1)

vp1.closeWindow()

# window should now close, the Plotter instance becomes unusable
# but mesh objects still exist in it:

printc("First Plotter actors:", vp1.actors, '\n press enter again')
# vp1.show()  # error here: window does not exist anymore. Cannot reopen.

##################################################################
# Can now create a brand new Plotter and show the old object in it
vp2 = Plotter(title='Second Plotter instance', pos=(500,0))
vp2.show(vp1.actors[0].color('red'))

##################################################################
# Create a third new Plotter and then close the second
vp3 = Plotter(title='Third Plotter instance')

vp2.closeWindow()
printc('vp2.closeWindow() called')

vp3.show(Hyperboloid()).close()

printc('done.')
