"""Trame integration example with interactive vedo updates."""
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

cone = vedo.Cone()
axes = vedo.Axes(cone).unpack()

plt = vedo.Plotter(offscreen=True)
plt += [cone, axes]

# -----------------------------------------------------------------------------
# Trame setup
# -----------------------------------------------------------------------------
server = get_server(client_type=client_type)
state, ctrl = server.state, server.controller

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------
@state.change("resolution")
def update_resolution(resolution, **kwargs):
    cone.color(resolution)
    ctrl.view_update()

def reset_resolution():
    cone.color("red5")
    ctrl.view_update()

# -----------------------------------------------------------------------------
# GUI
# -----------------------------------------------------------------------------
with SinglePageLayout(server, full_height=True) as layout:
    layout.title.set_text("Use slider to change color")

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

    with layout.toolbar:
        vuetify.VSpacer()
        vuetify.VSlider(
            v_model=("resolution", "blue5"),
            min=3,
            max=60,
            step=1,
            hide_details=True,
            dense=True,
            style="max-width: 300px",
        )
        with vuetify.VBtn(icon=True, click=reset_resolution):
            vuetify.VIcon("mdi-restore")

        vuetify.VDivider(vertical=True, classes="mx-2")

        vuetify.VSwitch(
            v_model="$vuetify.theme.dark",
            hide_details=True,
            dense=True,
        )
        with vuetify.VBtn(icon=True, click=ctrl.view_reset_camera):
            vuetify.VIcon("mdi-crop-free")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
server.start()
