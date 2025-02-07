
import numpy as np
from vedo import download, dataurl, precision, settings
from vedo import Volume, Point, Plotter, Axes


def extract_geoanomaly(input_vol, xind, yind) -> np.array:
    # variable to store output volume
    out_vol = np.zeros_like(input_vol, dtype=np.float32)
    # extract data corresponding to anomaly position in input volume
    sub_vol = input_vol[xind[0] : xind[1], yind[0] : yind[1], :].copy()
    # replace the section in the output volume corresponding to
    # the location of the extracted anomaly
    out_vol[xind[0] : xind[1], yind[0] : yind[1], :] = sub_vol
    return out_vol


def slider_isovalue(widget, _event, init_value=None):
    global prev_value

    value = init_value if init_value else widget.value

    # snap to the closest allowed value
    idx = (np.abs(allowed_vals - value)).argmin()
    value = allowed_vals[idx]
    prev_value = value

    value_name = precision(value, 2)

    if value_name in bacts:  # reusing the already existing mesh
        torender = bacts[value_name]

    else:  # else generate
        torender = [
            iso,
            vol1.isosurface(value).c("#afe1af").flat(),
            vol2.isosurface(value).c("#ffa500").flat(),
            vol3.isosurface(value).c("#8A8AFF").flat(),
            vol4.isosurface(value).c("#faa0a0").flat(),
        ]
        bacts.update({value_name: torender})  # store it

    for m in torender:
        m.name = "AnomalyIsoSurface"
    plt.remove("AnomalyIsoSurface").add(torender)

######################################################
# general settings
settings.default_font = "Roboto"

path = download(dataurl+"geo_dataset.npy")
dataset = np.load(path)

# invert the 'z-axis' in a way the shallow depth values
# are displayed on top in the rendered window
dataset = np.flip(dataset, axis=2)

min_value = np.nanmin(dataset)
rmin = min_value - 0.1

# replace NaNs with a value to mask them in the rendered window
nan_ind = np.isnan(dataset)
dataset[nan_ind] = 0

# generate a volume object
vol = Volume(dataset, spacing=[15, 15, 2])

# generate a surface object
iso = vol.isosurface(value=rmin).smooth()
iso.c("k5").alpha(0.15).lighting("off")


###########################
# get the central-western anomaly defined by:
X_ind = (1, 30)
Y_ind = (22, 46)
vol1 = extract_geoanomaly(dataset, X_ind, Y_ind)
vol1 = Volume(vol1, spacing=[15, 15, 2])

txt = "Central-western geoanomaly\n"
txt += "(mid to strong S-wave values)"
capt1 = Point((581, 207, 212)).caption(txt, size=(0.2, 0.05), justify='center-left', c="k")


###########################
# get the central geoanomaly defined by:
X_ind = (32, 41)
Y_ind = (26, 35)
vol2 = extract_geoanomaly(dataset, X_ind, Y_ind)
vol2 = Volume(vol2, spacing=[15, 15, 2])

txt = "Central geoanomaly\n"
txt += "(low to mid S-wave values)"
capt2 = Point([514, 475, 193]).caption(txt, size=(0.2, 0.05), justify='center-left', c="k")

###########################
# get the south-eastern geoanomaly defined by:
X_ind = (26, 53)
Y_ind = (1, 25)
vol3 = extract_geoanomaly(dataset, X_ind, Y_ind)
vol3 = Volume(vol3, spacing=[15, 15, 2])

txt = "Soth-eastern geoanomaly\n"
txt += "(mid to strong S-wave values)"
capt3 = Point([215, 500, 211]).caption(txt, size=(0.2, 0.05), justify='center-left', c="k")

###########################
# get the north-eastern geoanomaly defined by:
X_ind = (42, 56)
Y_ind = (37, 53)
vol4 = extract_geoanomaly(dataset, X_ind, Y_ind)
vol4 = Volume(vol4, spacing=[15, 15, 2])

txt = "North-eastern geoanomaly\n"
txt += "(mid to strong S-wave values)"
capt4 = Point([712, 630, 201]).caption(txt, size=(0.2, 0.05), justify='center-left', c="k")

###########################
# add slider options to rendered window based on code at
# https://github.com/marcomusy/vedo/blob/master/vedo/applications.py
scalar_range = [2.80, 3.60]
prev_value = 1e30
allowed_vals = np.array([2.80, 2.90, 3.00, 3.10, 3.20, 3.30, 3.40, 3.50, 3.60])
bacts = {}  # catch the meshes so we don't need to recompute

axs = Axes(
    iso,
    xtitle="Easting",
    xtitle_color="dr",
    xline_color="dr",
    xtitle_backface_color="w",
    ytitle="Northing",
    ytitle_color="dg",
    yline_color="dg",
    ytitle_backface_color="w",
    ztitle="Depth",
    ztitle_color="db",
    zline_color="db",
    ztitle_backface_color="w",
    text_scale=1.25,
    xygrid=False,
    z_inverted=True,
    xlabel_size=0.0,
    ylabel_size=0.0,
    zlabel_size=0.0,
)

plt = Plotter(size=(1400, 1200))
plt.add_slider(
    slider_isovalue,
    scalar_range[0], scalar_range[1], 
    value=scalar_range[0], 
    pos=4, title="value", show_value=True, delayed=False
)
plt.show(iso, axs, capt1, capt2, capt3, capt4, viewup="z", interactive=False)
slider_isovalue(None, None, init_value=scalar_range[0]) # init the first value
plt.interactive().close()
