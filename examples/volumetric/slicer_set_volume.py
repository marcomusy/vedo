"""Test swapping the input of Slicer3DPlotter without recreating the window.

Press space to toggle between two synthetic volumes.
Press x to preserve the current slice indices when switching volume.
Press z to reset slices to the new volume defaults."""

import numpy as np
from vedo import Text2D, Volume
from vedo.applications import Slicer3DPlotter


def make_volume(shape, center, sigma, scale=1.0, bias=0.0):
    x, y, z = np.mgrid[: shape[0], : shape[1], : shape[2]]
    dx = (x - center[0]) / sigma[0]
    dy = (y - center[1]) / sigma[1]
    dz = (z - center[2]) / sigma[2]
    g1 = np.exp(-(dx * dx + dy * dy + dz * dz))
    g2 = np.exp(-((dx + 1.2) ** 2 + (dy - 0.8) ** 2 + (dz + 0.5) ** 2) * 1.8)
    field = scale * (g1 - 0.35 * g2) + bias
    vol = Volume(field.astype(np.float32))
    return vol

def update_status():
    name, vol = volumes[state["index"]]
    dims = vol.dimensions().tolist()
    status.text(f"Volume: {name} | dims={dims}")

def swap_volume(reset_slices):
    state["index"] = 1 - state["index"]
    _, new_volume = volumes[state["index"]]
    plt.set_volume(new_volume, reset_slices=reset_slices)
    update_status()
    plt.render()

def on_key_press(evt):
    if evt.keypress == "space":
        swap_volume(reset_slices=False)
    elif evt.keypress == "x":
        swap_volume(reset_slices=False)
    elif evt.keypress == "z":
        swap_volume(reset_slices=True)


help_text = Text2D(
    __doc__,
    pos="top-left",
    s=0.8,
    bg="yellow",
    alpha=0.25,
)
status = Text2D("", pos="bottom-right", font="Calco", s=0.7, c="k")

state = {"index": 0}
volumes = [
    ("vol-A", make_volume((30, 50, 50), (25, 25, 25), (10, 10, 10), scale=1.0)),
    ("vol-B", make_volume((60, 60, 60), (30, 30, 30), (12, 12, 12), scale=1.5)),
]

plt = Slicer3DPlotter(
    volumes[0][1],
    cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
    bg="white",
    bg2="blue9",
)
plt.add_callback("key press", on_key_press)
update_status()
plt.show(help_text, status, viewup="z")
plt.close()
