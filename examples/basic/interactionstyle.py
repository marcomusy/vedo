"""
Scene interaction styles. Available styles are:
      - 0, TrackballCamera
      - 1, TrackballActor
      - 2, JoystickCamera
      - 3, Unicam
      - 4, Flight
      - 5, RubberBand3D
      - 6, RubberBandZoom
"""
print(__doc__)
from vedo import *

show(Spring(), Cube(), at=[0, 1], shape=(1, 3)).interactive()

t = Text2D(
    """TrackballCamera is the default
...lets change it to JoystickCamera:""",
    c="k", bg="w", s=0.8,
)

print("..change it to JoystickCamera")
show(Paraboloid(), t, at=2, mode="JoystickCamera").interactive()
