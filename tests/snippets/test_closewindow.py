"""Closing the Rendering Window

Press q:
Control returns to terminal,
window will not close but become unresponsive"""
from vedo import *

mesh = Paraboloid()

if settings.dry_run_mode:
    printc('vedo is in dry_run_mode, skipping window interaction test', c='tomato', invert=1)
    exit(0)

plt1 = show(mesh, __doc__, title='First Plotter instance')
# Now press 'q' to exit the window interaction,
# windows stays open but not reactive anymore.

# You can go back to interaction mode by simply calling:
#plt1.interactive()

printc('\nControl returned to terminal shell:', c='tomato', invert=1)
# ask('window is now unresponsive (press Enter here)', c='tomato', invert=1)

plt1.close()

# window should now close, the Plotter instance becomes unusable
# but mesh objects still exist in it:
printc("Objects in first Plotter:", len(plt1.objects), '\nPress q again')
# plt1.show()  # error here: window does not exist anymore. Cannot reopen.

# ##################################################################
# Can now create a brand new Plotter and show the old object in it
plt2 = Plotter(title='Second Plotter instance', pos=(500,0))
plt2.show(plt1.objects[0].color('red'))

##################################################################
# Create a third new Plotter and then close the second
plt3 = Plotter(title='Third Plotter instance')

plt2.close()
printc('plt2.close() called')

plt3.show(Hyperboloid()).close()

printc('done.')
