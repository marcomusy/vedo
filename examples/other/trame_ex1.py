#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# See more examples at:
# https://github.com/Kitware/trame-tutorial

from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk, vuetify

import vedo

cone = vedo.Cone()

plt = vedo.Plotter()
plt += cone

# -----------------------------------------------------------------------------
# Trame
# -----------------------------------------------------------------------------
server = get_server()

with SinglePageLayout(server) as layout:
    layout.title.set_text("Hello trame")

    with layout.content:

        with vuetify.VContainer(fluid=True, classes="pa-0 fill-height"):
            plt.reset_camera()
            view = vtk.VtkLocalView(plt.window)

server.start()
