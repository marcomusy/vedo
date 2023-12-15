"""Drag the red points to modify the wire path
Press "b" to compute the magnetic field"""
import numpy as np
from vedo import settings, mag, utils
from vedo import Arrows, Points, Axes, Plotter, Text2D, Circle

def func(evt):
    if evt.keypress != "b":
        return
    txt.text("..computing field in space, please wait!")
    txt.c('red5').background('yellow7')
    plt.render()

    pts = sptool.spline().vertices # extract the current spline
    field = []
    for probe in probes:
        B = np.zeros(3)
        for p0,p1 in zip(pts, np.roll(pts,1, axis=0)):
            p = (p0+p1)/2
            r = mag(p-probe)
            B += np.cross(p1-p0, p-probe)/r**3  # Biot-Savart law
        B /= max(1, mag(B))  # clamp the field magnitude near the wire
        field.append(B)
    field = np.array(field)

    arrows = Arrows(probes, probes+field/5).c('black')
    txt.text(__doc__).c('black').background(None)

    ppts1 = Points(probes)
    ppts1.pointdata["BField"] = field
    domain = ppts1.tovolume(n=4, dims=(10,10,10)) # interpolate

    ppts2 = ppts1.clone()  # make a copy
    ppts2.pointdata["BFieldIntensity"] = mag(field*255/3).astype(np.uint8)
    vol = ppts2.tovolume(n=4, dims=(10,10,10)).crop(back=0.5)
    isos = vol.isosurface(np.arange(0,250, 12)).smooth()
    isos.cmap("rainbow").lighting('off').alpha(0.5).add_scalarbar()
    isos.name = "Isosurfaces"

    streamlines = domain.compute_streamlines(
    	probes,
        max_propagation=0.5,
        initial_step_size=0.01,
        direction="both",
    )
    streamlines.c('black').linewidth(2)
    plt.remove("Arrows", "StreamLines", "Isosurfaces", "Axes")
    plt.add(arrows, streamlines, isos,  Axes(streamlines)).render()


probes = utils.pack_spheres([-2,2, -2,2, -2,2], radius=0.7)

settings.use_depth_peeling = True
settings.multi_samples = 0

plt = Plotter()
plt.add_callback("key press", func)

txt = Text2D(__doc__, font="Kanopus")
plt += txt

# Create a set of points in space to form a spline
circle = Circle(res=8)  # resolution = 8 points
sptool = plt.add_spline_tool(circle, pc='red', lw=4, closed=True)

plt.show().close()
