#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# See more examples at:
# https://github.com/Kitware/trame-tutorial

from trame.app import get_server
from trame.ui.vuetify import SinglePageLayout
from trame.widgets import vtk, vuetify

import vedo

sphere = vedo.Sphere().lw(1)
sphere.cmap("Spectral_r", sphere.vertices[:, 1])
axes = vedo.Axes(sphere)

plt = vedo.Plotter()
plt += sphere
plt += axes.unpack()
plt += vedo.Text3D("A color sphere", font='Quikhand', s=0.2, pos=[-1,1,-1])

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
