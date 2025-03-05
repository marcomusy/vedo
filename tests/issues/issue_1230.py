"Test line.eval() and line.find_index_at_position()"
import numpy as np
import vedo


def slider_func(sld, event):
    eval_length = sld.value
    pe = line.eval(eval_length)
    idx_fraction = line.find_index_at_position(pe)
    idx_before = np.floor(idx_fraction).astype(int)
    idx_after = np.ceil(idx_fraction).astype(int) % len(points)
    ps = vedo.Points(points[[idx_before, idx_after]], c="red5", r=15)
    pe = vedo.Point(pe, c="green5", r=20)
    ps.name = "slidingpoints"
    pe.name = "slidingpoints"
    txt.text(f"index_at_position(): {idx_fraction:.3f}")
    plt.remove("slidingpoints").add(ps, pe).render()


n = 10
endpt = 1
angles = np.linspace(0, 2 * np.pi, n, endpoint=endpt)
radii = np.linspace(0.5, 1, n, endpoint=endpt)
points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

vedo.settings.default_font = "Roboto"
line = vedo.Line(points, closed=endpt, lw=2).split_polylines()
labs = line.labels2d()
txt = vedo.Text2D(pos="bottom-left", c="k", bg="y")

plt = vedo.Plotter()
plt.add_slider(slider_func, 0, 0.99999, value=0, title="Fractional position")
plt.show(line, labs, txt, __doc__)
