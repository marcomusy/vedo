"""Record and playback camera movements and other events
\rightarrow Move the cube around, press 1, and finally press q"""
from vedo import Cube, Plotter

plt1 = Plotter(axes=1, interactive=0, title="recording window")
evts = plt1.show(Cube(), __doc__).record()
# print("Events:", evts) # a simple string (also saved as .vedo_recorded_events.log)

plt2 = Plotter(axes=1, interactive=0, title="playback window", pos=(1100,0))
plt2.show(Cube(), "...now playing!").play(evts).interactive().close()
