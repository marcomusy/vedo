#!/usr/bin/env python3
#
"""Draw molecular streamlines from a Gaussian cube file."""
from vedo import download, loadGaussianCube, Sphere, Tubes, show
from vedo.applications.chemistry import Molecule


cube_path = download("https://www.dropbox.com/scl/fi/2927evbr9s44sgs5y4iwl/crownK.pot.cube?rlkey=as6xdseh2vqo5pn00pym1414v&st=2d63gloi")
poly, volume = loadGaussianCube(cube_path, b_scale=20, hb_scale=20)
atom_positions = poly.vertices

molecule = Molecule(poly)
molecule.use_ball_and_stick()
molecule.use_multi_cylinders_for_bonds(False)
molecule.set_atom_radius_scale(0.75)
molecule.set_bond_radius(0.5)
molecule.set_bond_color_mode("single")
molecule.set_bond_color("k5")

center = atom_positions[0]
seed_domain = Sphere(pos=center, r=1.0)
seeds = seed_domain.clone().subsample(0.1)

# Use the gradient of the electron density as the vector field.
volume.pointdata["Gradient"] = volume.gradient()
streamlines = volume.compute_streamlines(
    seeds,
    direction="backward",
    integrator="rk4",
    initial_step_size=0.05,
    max_propagation=500,
)
streamtubes = Tubes(streamlines, r=0.10, res=8).cmap("Spectral_r", vmax=1)
streamtubes.add_scalarbar("E field")
show(molecule, streamtubes, __doc__, axes=1).close()
