
from vedo import *
from vedo.pyplot import histogram


def set_mask_by_thresholds(thresholds):
    vol_arrc = np.zeros_like(vol_arr, dtype=np.uint8)
    vol_arrc[(vol_arr > thresholds[0]) & (vol_arr < thresholds[1])] = 1
    vol.mask(vol_arrc)

def slider1(w, e):
    if slid1.value > slid2.value:
        slid1.value = slid2.value
    set_mask_by_thresholds([slid1.value, slid2.value])

def slider2(w, e):
    if slid2.value < slid1.value:
        slid2.value = slid1.value
    set_mask_by_thresholds([slid1.value, slid2.value])


vol = Volume(dataurl+"embryo.slc")
vol.mapper = "gpu"
vol.cmap("rainbow").alpha([0, 0.1, 0.2, 0.3, 0.4, 0.8, 1])
vol_arr = vol.tonumpy()

histo = histogram(vol, bins=25, c="rainbow", logscale=True, ytitle="")
histo = histo.clone2d(size=0.5)

plt = Plotter(axes=7)
rng = vol.scalar_range()

slid2 = plt.add_slider(
    slider2,
    xmin=rng[0],
    xmax=rng[1],
    value=rng[1],
    slider_length=0.02,
    slider_width=0.06,
    alpha=0.75,
    c="red2",
    delayed=True, # update only when mouse is released
)
slid1 = plt.add_slider(
    slider1,
    xmin=rng[0],
    xmax=rng[1],
    value=rng[0],
    slider_length=0.01,
    slider_width=0.05,
    alpha=0.75,
    tube_width=0.0015,
    c="blue2",
    delayed=True,
)

plt.show(vol, histo)
plt.close()