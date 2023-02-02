#!/usr/bin/env python
#
from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk, vuetify

from vedo import Volume, Axes, Plotter, dataurl

vol = Volume(dataurl+"embryo.slc")

plt = Plotter(bg='Wheat')
plt += [vol, Axes(vol)]
plt += vol.isosurface().shift(300,0,0)

# ------------------------------------------------------------
# Web Application setup
# ------------------------------------------------------------
server = get_server()
ctrl = server.controller

with SinglePageLayout(server) as layout:
    layout.title.set_text("Hello trame")

    with layout.content:
        with vuetify.VContainer(
            fluid=True,
            classes="pa-0 fill-height",
        ):
            plt.reset_camera()
            view = vtk.VtkRemoteView(plt.window)
            # view = vtk.VtkLocalView(plt.window)

server.start()
