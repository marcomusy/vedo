"""Press TAB to toggle active panel and freeze the other"""
from vedo import settings, Cube, Image, dataurl, RendererFrame, Plotter
from vedo.interactor_modes import MousePan

settings.enable_default_keyboard_callbacks = False
settings.default_font = "Roboto"

active = 0
inactive = 1

cube = Cube().rotate_x(10)
img = Image(dataurl+"images/dog.jpg") 


def toggle_active(event):
    global active, inactive
    if event.keypress == "Tab":  # toggle active renderer
        active, inactive = inactive, active
        plt.at(active).user_mode(modes[active])
        plt.at(inactive).remove(frames[inactive]).freeze(True)
        plt.at(active).add(frames[active]).freeze(False)
        plt.render()
    elif event.keypress == "q":
        plt.close()

frame0 = RendererFrame(lw=10, c="red5", alpha=1)
frame1 = RendererFrame(lw=10, c="red5", alpha=1)

plt = Plotter(shape=(1,2), sharecam=False, axes=1)
modes = [0, MousePan()]
frames = [frame0, frame1]

plt.at(0).add(cube, frame0, __doc__).reset_camera()
plt.at(1).add(img)
plt.add_callback('key press', toggle_active)
plt.at(inactive).freeze()
plt.show(interactive=True).close()