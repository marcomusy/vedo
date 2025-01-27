from vedo import *


def render_slice(vslice, name):
    vslice.cut_with_scalar(rmin, "input_scalars", invert=True)
    vslice.triangulate()
    vslice.cmap(cmap_slicer, vmin=rmin, vmax=rmax).lighting("off")
    isos = vslice.isolines(vmin=rmin, vmax=rmax, n=12).c("black")
    vslice.name = name
    isos.name = name
    plt.remove(name).add(vslice, isos)

def slider_function_x(widget, event):
    i = int(widget.value)
    if i == widget.previous_value:
        return
    widget.previous_value = i
    render_slice(vol.xslice(i), "XSlice")

def slider_function_y(widget, event):
    j = int(widget.value)
    if j == widget.previous_value:
        return
    widget.previous_value = j
    render_slice(vol.yslice(j), "YSlice")


def slider_function_z(widget, event):
    k = int(widget.value)
    if k == widget.previous_value:
        return
    widget.previous_value = k
    render_slice(vol.zslice(k), "ZSlice")


if __name__ == "__main__":

    settings.default_font = "Roboto"
    cmap_slicer = "RdBu"

    datapath = download(dataurl+"geo_dataset.npy")
    dataset = np.load(datapath)
    min_value = np.nanmin(dataset)
    max_value = np.nanmax(dataset)

    rmin = np.nanquantile(dataset, q=0.30)
    rmax = np.nanquantile(dataset, q=0.95)

    # replace NaNs with a value to mask them in the rendered window
    nan_ind = np.isnan(dataset)
    dataset[nan_ind] = 0

    vol = Volume(dataset, spacing=[15, 15, 2])
    dims = vol.dimensions()

    iso = vol.isosurface(rmin).smooth()
    iso.cmap(cmap_slicer, vmin=min_value, vmax=max_value)
    iso.add_scalarbar3d(c="black", title="scalar value")
    iso.scalarbar = iso.scalarbar.clone2d("center-right", size=0.2)
    iso.c("k5").alpha(0.1).lighting("off").wireframe().pickable(False).backface_culling()

    plt = Plotter(size=(1400, 1200))

    plt.add_slider(
        slider_function_x,
        0, dims[0],
        pos=[(0.7, 0.12), (0.95, 0.12)],
        show_value=False,
        c="dr",
    )

    plt.add_slider(
        slider_function_y,
        0, dims[1],
        pos=[(0.7, 0.08), (0.95, 0.08)],
        show_value=False,
        c="dg",
    )

    plt.add_slider(
        slider_function_z,
        0, dims[2],
        pos=[(0.7, 0.04), (0.95, 0.04)],
        show_value=False,
        c="db",
    )

    plt.show(iso, viewup="z", axes=1).close()
