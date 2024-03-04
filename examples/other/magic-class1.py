"""Use magicclass to plot\nrandom points and a histogram."""
import numpy as np
import vedo
try:
    from magicclass import magicclass, field
    from magicclass.ext.vtk import VedoCanvas
except ImportError:
    print("Please install magicclass with: pip install magic-class")

@magicclass
class ViewerUI:
    canvas = field(VedoCanvas)

    def plot_random_points(self):
        """Plot random data."""
        # create a points object and a set of axes
        coords = np.random.randn(1000, 3)
        data = np.cos(coords[:,1])
        points = vedo.Points(coords)
        points.cmap("viridis", data).add_scalarbar3d()
        axes = vedo.Axes(points, c="white")
        # create a histogram of data
        histo = vedo.pyplot.histogram(
            data, c="viridis", title=" ", xtitle="", ytitle="",
        )
        histo = histo.clone2d("bottom-right", size=0.25)
        ui.canvas.plotter.remove("Axes", "Points", "Histogram1D")
        ui.canvas.plotter.add(points, axes, histo)
        ui.canvas.plotter.reset_camera().render()

if __name__ == "__main__":
    ui = ViewerUI()
    ui.canvas.plotter.add(__doc__)
    ui.show()
