"""Create a simple Play/Pause app with a timer event
You can interact with the scene during the loop!
..press q to quit"""
import vedo


def buttonfunc():
    global timerId
    plotter.timerCallback("destroy", timerId)
    if "Play" in button.status():
        # instruct to call handle_timer() every 10 msec:
        timerId = plotter.timerCallback("create", dt=10)
    button.switch()

def handle_timer(event):
    ### Animate your stuff here ######################################
    earth.rotateZ(1)            # rotate the Earth by 1 deg
    plotter.render()


plotter = vedo.Plotter()

timerId = None
button = plotter.addButton(buttonfunc, states=[" Play ","Pause"], size=40)
evntId = plotter.addCallback("timer", handle_timer)

earth = vedo.Earth()

plotter.show(earth, __doc__, axes=1, bg2='b9', viewup='z').close()
