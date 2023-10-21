"""Create 2 independent timer callbacks:"""
from vedo import *

# Defining a function to be called by a timer event
def func1(event):
    # Check if this function was called by the right timer
    if event.timerid != ida:
        return
    # Rotate a cube mesh and set its color to green5
    msh.rotate_z(1.0).c("green5")
    # Update text and print a message with the event and timer ids
    txt.text("func1() called").background('green5')
    printc(f"func1() id={event.id}, timerid={event.timerid}", c='g')
    plt.render()

# Defining another function to be called by a different timer event
def func2(event):
    # Check if this function was called by the right timer
    if event.timerid != idb:
        return
    # Rotate the same cube mesh in a different direction
    msh.rotate_x(5.0).c("red5")
    # Update text and print a message with the event and timer ids
    txt.text("func2() called").background('red5')
    printc(f"func2() id={event.id}, timerid={event.timerid}", c='r')
    plt.render()

# Create a cube mesh and a text object
msh = Cube()
txt = Text2D(font="Calco", pos='top-right')

# Create a plotter object with axes
plt = Plotter(axes=1)
# plt.initialize_interactor() # on windows this is needed

# Add the two callback functions to the plotter's timer events
id1 = plt.add_callback("timer", func1)
id2 = plt.add_callback("timer", func2)
printc("Creating Timer Callbacks with IDs:", id1, id2)

# Start two timers, one with a delay of 1s and the other with a delay of 2.3s
ida = plt.timer_callback("start", dt=1000)
idb = plt.timer_callback("start", dt=2300)
printc("Starting timers with IDs         :", ida, idb)

# Stop the first timer using its ID
# plt.timer_callback("stop", ida)

plt.show(msh, txt, __doc__, viewup='z')
plt.close()
