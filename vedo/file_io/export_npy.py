from __future__ import annotations
"""NPY/NPZ scene export helpers."""

import numpy as np

import vedo

__docformat__ = "google"


def _export_npy(plt, fileoutput="scene.npz") -> None:
    """Export the current scene to a NPY or NPZ file."""
    from .scene import _plotter_to_scene_dict

    fileoutput = str(fileoutput)
    if plt is None:
        vedo.logger.error("_export_npy(): no active Plotter found")
        return

    sdict = _plotter_to_scene_dict(plt)

    if fileoutput.endswith(".npz"):
        np.savez_compressed(fileoutput, vedo_scenes=[sdict])
    else:
        np.save(fileoutput, [sdict])
