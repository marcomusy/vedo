from vedo import show, download
from vedo.chemistry import Molecule, PeriodicTable, Protein

###################################################################
# Create an instance of PeriodicTable
pt = PeriodicTable()
print(pt)

# Get atomic number from symbol
symbol = "Co"
print(f"\nAtomic number of {symbol}: {pt.get_atomic_number(symbol)}")

path1 = download("https://www.dropbox.com/scl/fi/6ml4xr88xyyaupe8hs61u/3gea.pdb?rlkey=xm8lkknyzklqacepqmqmod0i0&dl=0")
path2 = download("https://www.dropbox.com/scl/fi/91tcay1a8vr8r5nogudr6/caffeine.pdb?rlkey=hhw217vyje54z6b9tdjzurmoq&dl=0")

###################################################################
protein = Protein(path1)
protein.set_coil_width(0.3)
protein.set_helix_width(0.8)
protein.set_sphere_resolution(15)
show(protein, axes=1).close()

###################################################################
mol = Molecule(path2)

# mol = Molecule()
# o_atom = mol.append_atom([0, 0, 0], 8)  # Oxygen
# h1_atom = mol.append_atom([0.757, 0.586, 0], 1)  # Hydrogen 1
# h2_atom = mol.append_atom([-0.757, 0.586, 0], 1)  # Hydrogen 2
# mol.append_bond(o_atom, h1_atom, 1)  # Single bond O-H1
# mol.append_bond(o_atom, h2_atom, 1)  # Single bond O-H2

# Display atom info
print(f"Number of atoms: {mol.get_number_of_atoms()}")
print(f"Number of bonds: {mol.get_number_of_bonds()}")
print(f"Atom positions:\n{mol.get_atom_positions()}")
print(f"Atomic numbers: {set(mol.get_atomic_numbers())}")
print(f"Bond 0: {mol.get_bond(0)}")

# Customize rendering
mol.use_ball_and_stick()

show(mol, axes=1)
