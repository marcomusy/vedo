"""Scene interaction styles"""
from vedo import Text2D, Cube, Paraboloid, Plotter
from vedo.plotter.modes import MousePan

# Shared on-screen instructions while switching interaction styles.
msg = Text2D(
    """TrackballCamera is the default mode\n(press q to proceed)""",
    c="k", bg="yellow7", s=1.2,
)

plt = Plotter(shape=(1,2))
plt.at(0).show(Cube(), msg).interactive()

msg.text("..lets change it to JoystickCamera").background("indigo7")
plt.at(1).show(Paraboloid(), mode="JoystickCamera").interactive()

msg.text("..lets change it again to MousePan").background("red6")
mode = MousePan()
plt.user_mode(mode).interactive()

plt.close()
