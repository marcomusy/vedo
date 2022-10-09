"""Drag the red points to modify the current path
Press b to compute the magnetic field
(field streamlines are approximate)"""
from vedo import *

def func(evt):
    if evt.keyPressed != "b":
        return
    txt.text("..computing field in space, please wait!").c('red')
    plt.render()

    pts = sptool.spline().points() # extract the current spline
    field = []
    for probe in probes:
        B = np.zeros(3)
        for p0,p1 in zip(pts, np.roll(pts,1, axis=0)):
            p = (p0+p1)/2
            r = mag(p-probe)
            B += np.cross(p1-p0, p-probe)/r**3  # Biot-Savart law
        B /= max(1, mag(B))  # clamp the field magnitude
        field.append(B)
    field = np.array(field)

    arrows = Arrows(probes, probes+field/5).c('black')
    txt.text(__doc__).c('black')

    ppts = Points(probes)
    ppts.pointdata["B"] = field
    domain = ppts.tovolume(N=4, dims=(50,50,50)) # interpolate

    streamlines = StreamLines(
    	domain,
    	probes,
        maxPropagation=0.5,
        initialStepSize=0.01,
        direction="both",
    )
    streamlines.c('black').lw(2)
    plt.remove("Arrows", "StreamLines", "Axes")
    plt.add(arrows, streamlines, Axes(streamlines))

probes = utils.pack_spheres([-2,2, -2,2, -2,2], radius=0.75)

plt = Plotter()
plt.add_callback("key press", func)

txt = Text2D(__doc__)
plt += txt

# Create a set of points in space to form a spline
circle = Circle(res=8) # resolution = 8 points
plt.add_spline_tool(circle, pc='red', lw=4, closed=True)

plt.show().close()
