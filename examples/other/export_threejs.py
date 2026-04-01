"""Export a vedo scene as a
standalone Three.js webpage."""

from pathlib import Path

from vedo import Cube, Mesh, Plotter, Text3D, dataurl


plt = Plotter(size=(900, 700), bg="ivory", bg2="lightblue")

tex_cube = Cube().texture(dataurl + "textures/wood1.jpg")
tex_cube.rotate_x(20).rotate_z(20).x(-0.8)

plain_cube = Cube().c("tomato").alpha(0.7).x(0.9)

bunny = Mesh(dataurl + "bunny.obj")
bunny.normalize().scale(0.55).rotate_x(90).pos(1, 0.1, 0.25)
bunny.c("gold").lighting("glossy")

title = Text3D(__doc__, s=0.08, depth=0.02, font="Edo", c="k")
title.rotate_x(90).pos(-1, 1.1, 1)

plt.show(tex_cube, plain_cube, bunny, title, axes=1, viewup="z", zoom=1.2)

threejs_options = {
    "headlight_intensity": 1.0,
    "ambient_scale": 0.25,
    "specular_scale": 0.45,
    "fallback_specular_strength": 0.35,
    "fallback_shininess": 28.0,
    "preserve_base_color": False,
    "pack_arrays": True,
}

# This exports a self-contained HTML page that reconstructs the scene with Three.js.
outfile = Path(__file__).with_suffix(".html")
plt.export(str(outfile), backend="threejs", backend_options=threejs_options)

print(f"Type:\n firefox {outfile}")
