"""Trame integration example with a volume scene."""
#!/usr/bin/env python
#
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

from vedo import Volume, Axes, Plotter, dataurl

vol = Volume(dataurl+"embryo.slc")

plt = Plotter(bg='Wheat', offscreen=True)
plt += [vol, Axes(vol)]
plt += vol.isosurface().shift(300,0,0)

# ------------------------------------------------------------
# Web Application setup
# ------------------------------------------------------------
server = get_server(client_type=client_type)
ctrl = server.controller

with SinglePageLayout(server, full_height=True) as layout:
    layout.title.set_text("Hello trame")

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            plt.reset_camera()
            view = vtk.VtkRemoteView(plt.window, interactive_ratio=1)
            ctrl.view_update = view.update
            ctrl.view_reset_camera = view.reset_camera
            ctrl.view_update()

server.start()
