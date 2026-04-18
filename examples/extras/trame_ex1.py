#!/usr/bin/env python3
"""Trame integration example with a simple vedo scene."""

from importlib import import_module

from trame.app import get_server

try:
    try:
        SinglePageLayout = import_module("trame.ui.vuetify3").SinglePageLayout
        vtk = import_module("trame.widgets.vtk")
        vuetify = import_module("trame.widgets.vuetify3")
        client_type = "vue3"
    except ImportError:
        SinglePageLayout = import_module("trame.ui.vuetify").SinglePageLayout
        vtk = import_module("trame.widgets.vtk")
        vuetify = import_module("trame.widgets.vuetify")
        client_type = "vue2"
except ImportError as exc:
    raise SystemExit(
        "This example requires trame widget packages. Install with:\n"
        "> pip install trame trame-vtk trame-vuetify"
    ) from exc

import vedo

sphere = vedo.Sphere().lw(1)
sphere.cmap("Spectral_r", sphere.vertices[:, 1])
axes = vedo.Axes(sphere)

plt = vedo.Plotter(offscreen=True)
plt += sphere
plt += axes.unpack()
plt += vedo.Text3D("A color sphere", font="Quikhand", s=0.2, pos=[-1, 1, -1])

server = get_server(client_type=client_type)
ctrl = server.controller

with SinglePageLayout(server, full_height=True) as layout:
    layout.title.set_text("Hello trame")

    with layout.content:
        with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
            plt.reset_camera()
            # RemoteView is the most practical path for an existing vedo render window.
            view = vtk.VtkRemoteView(plt.window, interactive_ratio=1)
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera
            ctrl.view_update()

server.start()
