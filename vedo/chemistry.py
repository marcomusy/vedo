import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkMolecule
from vtkmodules.vtkFiltersCore import vtkMoleculeAppend
from vtkmodules.vtkIOChemistry import vtkPDBReader
from vtkmodules.vtkRenderingCore import vtkActor, vtkPolyDataMapper
from vtkmodules.vtkDomainsChemistry import vtkPeriodicTable, vtkMoleculeMapper
from vtkmodules.vtkDomainsChemistry import vtkProteinRibbonFilter
from vtkmodules.vtkCommonCore import vtkIdList
from vtkmodules.vtkCommonDataModel import vtkPolyData


class PeriodicTable:
    """
    A Vedo-compatible class for accessing periodic table data, wrapping vtkPeriodicTable.

    This class provides access to element properties such as atomic numbers, names,
    symbols, and covalent radii, using VTK's built-in periodic table database.

    Attributes:
        periodic_table (vtkPeriodicTable): The underlying VTK periodic table object.
    """

    def __init__(self):
        """
        Initialize the PeriodicTable with VTK's built-in periodic table data.
        """
        self.periodic_table = vtkPeriodicTable()

    def get_element_name(self, atomic_number):
        """
        Get the name of the element with the given atomic number.

        Arguments:
            atomic_number (int): The atomic number of the element.

        Returns:
            str: The name of the element.
        """
        return self.periodic_table.GetElementName(atomic_number)

    def get_element_symbol(self, atomic_number):
        """
        Get the symbol of the element with the given atomic number.

        Arguments:
            atomic_number (int): The atomic number of the element.

        Returns:
            str: The symbol of the element.
        """
        return self.periodic_table.GetSymbol(atomic_number)

    def get_atomic_number(self, symbol):
        """
        Get the atomic number of the element with the given symbol.

        Arguments:
            symbol (str): The symbol of the element.

        Returns:
            int: The atomic number of the element.
        """
        return self.periodic_table.GetAtomicNumber(symbol)

    def get_covalent_radius(self, atomic_number):
        """
        Get the covalent radius of the element with the given atomic number.

        Arguments:
            atomic_number (int): The atomic number of the element.

        Returns:
            float: The covalent radius of the element.
        """
        return self.periodic_table.GetCovalentRadius(atomic_number)

    def get_vdw_radius(self, atomic_number):
        """
        Get the Van der Waals radius of the element with the given atomic number.

        Arguments:
            atomic_number (int): The atomic number of the element.

        Returns:
            float: The Van der Waals radius of the element.
        """
        return self.periodic_table.GetVDWRadius(atomic_number)

    def get_number_of_elements(self):
        """
        Get the total number of elements in the periodic table.

        Returns:
            int: The number of elements.
        """
        return self.periodic_table.GetNumberOfElements()

    def get_element_data(self, atomic_number):
        """
        Get all data for the element with the given atomic number.

        Arguments:
            atomic_number (int): The atomic number of the element.

        Returns:
            dict: A dictionary containing the element's name, symbol, and radii.
        """
        return {
            "name": self.get_element_name(atomic_number),
            "symbol": self.get_element_symbol(atomic_number),
            "covalent_radius": self.get_covalent_radius(atomic_number),
            "vdw_radius": self.get_vdw_radius(atomic_number),
        }

    def __getitem__(self, atomic_number):
        """
        Get element data by atomic number.

        Arguments:
            atomic_number (int): The atomic number of the element.

        Returns:
            dict: A dictionary containing the element's name, symbol, and radii.
        """
        return self.get_element_data(atomic_number)

    def __len__(self):
        """
        Get the number of elements in the periodic table.

        Returns:
            int: The number of elements.
        """
        return self.get_number_of_elements()

    def __iter__(self):
        """
        Iterate over all elements in the periodic table.

        Yields:
            tuple: (atomic_number, element_data) for each element.
        """
        for atomic_number in range(1, self.get_number_of_elements() + 1):
            yield atomic_number, self.get_element_data(atomic_number)

    def __contains__(self, atomic_number):
        """
        Check if an atomic number is in the periodic table.

        Arguments:
            atomic_number (int): The atomic number to check.

        Returns:
            bool: True if the atomic number exists, False otherwise.
        """
        return 1 <= atomic_number <= self.get_number_of_elements()

    def __str__(self):
        """Print info about the periodic table."""
        from vedo.colors import printc
        module = self.__class__.__module__
        name = self.__class__.__name__
        out = printc(
            f"{module}.{name} at ({hex(id(self))})".ljust(75),
            bold=True, invert=True, return_string=True,
        )
        out += "\x1b[0m"
        n = self.get_number_of_elements()
        out += f"Number of elements  : {n}\n"
        # Example usage of the periodic table
        atomic_number = 12
        out += f"Atomic number       : {atomic_number} (example entry)\n"
        out += f"Element name        : {self.get_element_name(atomic_number)}\n"
        out += f"Element symbol      : {self.get_element_symbol(atomic_number)}\n"
        out += f"Covalent radius     : {self.get_covalent_radius(atomic_number)}\n"
        out += f"Van der Waals radius: {self.get_vdw_radius(atomic_number)}\n"
        return out.rstrip() + "\x1b[0m"



def append_molecules(molecules):
    """
    Append multiple molecules into a single molecule.

    This function takes a list of Molecule objects and returns a new Molecule
    object that combines all atoms and bonds from the input molecules.

    Arguments:
        molecules (list of Molecule): The molecules to append.

    Returns:
        Molecule: A new Molecule object containing all atoms and bonds from the input molecules.
    """
    if not molecules:
        raise ValueError("No molecules provided to append.")

    # Create an instance of vtkMoleculeAppend
    append_filter = vtkMoleculeAppend()

    # Add each molecule's vtkMolecule to the append filter
    for mol in molecules:
        append_filter.AddInputData(mol.molecule)

    # Update the filter to generate the combined molecule
    append_filter.Update()

    # Get the output molecule from the filter
    combined_vtk_molecule = append_filter.GetOutput()

    # Create a new Molecule object
    combined_molecule = Molecule()

    # Set the combined vtkMolecule to the new Molecule object
    combined_molecule.molecule = combined_vtk_molecule

    # Reconfigure the mapper and actor
    combined_molecule.mapper.SetInputData(combined_vtk_molecule)
    combined_molecule.actor.SetMapper(combined_molecule.mapper)

    # Optionally, copy rendering settings from the first molecule
    if molecules:
        first_mol = molecules[0]
        if first_mol.mapper.GetRenderAtoms():
            combined_molecule.use_ball_and_stick()
        else:
            combined_molecule.use_space_filling()
        combined_molecule.set_atom_radius_scale(
            first_mol.mapper.GetAtomicRadiusScaleFactor()
        )
        combined_molecule.set_bond_radius(first_mol.mapper.GetBondRadius())

    return combined_molecule


class Atom:
    """
    A class representing an atom in a molecule, fully wrapping vtkAtom.

    Provides access to all methods and properties of vtkAtom as documented in:
    https://vtk.org/doc/nightly/html/classvtkAtom.html
    """

    def __init__(self, molecule, atom_id):
        """
        Initialize the Atom with a reference to the molecule and its ID.

        Arguments:
            molecule (vtkMolecule): The molecule containing this atom.
            atom_id (int): The ID of the atom in the molecule.
        """
        self.molecule = molecule
        self.atom_id = atom_id
        self.vtk_atom = self.molecule.GetAtom(self.atom_id)

    def get_atomic_number(self):
        """Get the atomic number of the atom.

        Returns:
            int: The atomic number.
        """
        return self.vtk_atom.GetAtomicNumber()

    def set_atomic_number(self, atomic_number):
        """Set the atomic number of the atom.

        Arguments:
            atomic_number (int): The new atomic number.
        """
        self.vtk_atom.SetAtomicNumber(atomic_number)

    def get_position(self):
        """Get the position of the atom as a NumPy array.

        Returns:
            np.ndarray: Array of shape (3,) with [x, y, z] coordinates.
        """
        pos = self.vtk_atom.GetPosition()
        return np.array([pos[0], pos[1], pos[2]])

    def set_position(self, position):
        """Set the position of the atom.

        Arguments:
            position (list or np.ndarray): The new [x, y, z] coordinates.
        """
        pos = np.asarray(position, dtype=float)
        self.vtk_atom.SetPosition(pos[0], pos[1], pos[2])

    def get_atom_id(self):
        """Get the ID of this atom.

        Returns:
            int: The atom's ID within the molecule.
        """
        return self.atom_id

    def get_molecule(self):
        """Get the molecule this atom belongs to.

        Returns:
            vtkMolecule: The parent molecule.
        """
        return self.molecule

    def __repr__(self):
        return f"Atom(ID={self.atom_id}, AtomicNumber={self.atomic_number}, Position={self.position})"


class Molecule:
    def __init__(self, pdb_file=None):
        # Create an empty molecule
        self.molecule = vtkMolecule()

        # Configure the mapper and actor for rendering
        self.mapper = vtkMoleculeMapper()
        self.actor = vtkActor()
        self.property = self.actor.GetProperty()

        if pdb_file:
            # Create and configure the PDB reader
            reader = vtkPDBReader()
            reader.SetFileName(pdb_file)
            reader.Update()

            # Get the PDB data
            pdb_data = reader.GetOutput()
            # print the point data available
            # print(pdb_data.GetPointData())
            # Array 0 name = atom_type
            # Array 1 name = atom_types
            # Array 2 name = residue
            # Array 3 name = chain
            # Array 4 name = secondary_structures
            # Array 5 name = secondary_structures_begin
            # Array 6 name = secondary_structures_end
            # Array 7 name = ishetatm
            # Array 8 name = model
            # Array 9 name = rgb_colors
            # Array 10 name = radius
            # for i in range(pdb_data.GetPointData().GetNumberOfArrays()):
            #     print(pdb_data.GetPointData().GetArray(i))
            points = pdb_data.GetPoints()
            point_data = pdb_data.GetPointData()

            # Extract atom information and add to molecule
            for i in range(points.GetNumberOfPoints()):
                position = points.GetPoint(i)
                # Default to Carbon if atomic number is not available
                atomic_number = 6
                if point_data.GetScalars("atom_type"):
                    atomic_number = int(point_data.GetScalars("atom_type").GetValue(i))
                # Add atom to molecule
                # if point_data.GetScalars("rgb_colors"):
                #     color = point_data.GetScalars("rgb_colors").GetTuple3(i)
                #     self.actor.GetProperty().SetColor(color)
                self.molecule.AppendAtom(atomic_number, position)

            # Add bonds if available
            if pdb_data.GetLines():
                lines = pdb_data.GetLines()
                lines.InitTraversal()
                id_list = vtkIdList()
                while lines.GetNextCell(id_list):
                    if id_list.GetNumberOfIds() == 2:
                        self.molecule.AppendBond(
                            id_list.GetId(0),
                            id_list.GetId(1),
                            1,  # Default to single bond
                        )

        # Set the molecule as input to the mapper
        self.mapper.SetInputData(self.molecule)
        self.actor.SetMapper(self.mapper)

        # Apply default rendering style
        self.use_ball_and_stick()

    def append_atom(self, position=None, atomic_number=6):
        """Add an atom to the molecule with optional position and atomic number.

        Arguments:
            position (list or np.ndarray, optional): [x, y, z] coordinates. Defaults to [0, 0, 0].
            atomic_number (int, optional): Atomic number (e.g., 6 for Carbon). Defaults to 6.

        Returns:
            Atom: The added atom object.
        """
        if position is None:
            position = [0, 0, 0]
        pos = np.asarray(position, dtype=float)
        vtk_atom = self.molecule.AppendAtom(atomic_number, pos[0], pos[1], pos[2])
        atom_id = (
            self.molecule.GetNumberOfAtoms() - 1
        )  # The ID will be the index of the last added atom
        return Atom(self.molecule, atom_id)

    def get_atom(self, atom_id):
        """Retrieve an atom by its ID.

        Arguments:
            atom_id (int): The ID of the atom.

        Returns:
            Atom: The atom object.
        """
        if atom_id >= self.get_number_of_atoms():
            raise ValueError(f"Atom ID {atom_id} exceeds number of atoms.")
        return Atom(self.molecule, atom_id)

    def remove_atom(self, atom_id):
        """Remove an atom by its ID.

        Arguments:
            atom_id (int): The ID of the atom to remove.
        """
        if atom_id >= self.get_number_of_atoms():
            raise ValueError(f"Atom ID {atom_id} exceeds number of atoms.")
        self.molecule.RemoveAtom(atom_id)
        # Update the mapper to reflect the changes
        self.mapper.SetInputData(self.molecule)
        self.mapper.Update()
        # Update the actor to reflect the changes
        self.actor.SetMapper(self.mapper)
        self.actor.Update()
        # Update the property to reflect the changes
        self.property.SetColor(1.0, 0.8, 0.6)  # Reset to default color
        self.property.SetOpacity(1.0)

    def get_array(self, name):
        """Get a point data array by name.

        The following arrays are available:

        - atom_type: Atomic number of the atom.
        - atom_types: Atomic type of the atom.
        - residue: Residue name.
        - chain: Chain identifier.
        - secondary_structures: Secondary structure type.
        - secondary_structures_begin: Start index of the secondary structure.
        - secondary_structures_end: End index of the secondary structure.
        - ishetatm: Is the atom a heteroatom?
        - model: Model number.
        - rgb_colors: RGB color of the atom.
        - radius: Radius of the atom.

        Arguments:
            name (str): The name of the array.

        Returns:
            np.ndarray: The array data.
        """
        if not self.molecule.GetPointData().HasArray(name):
            raise ValueError(f"Array '{name}' not found in molecule.")
        return vtk_to_numpy(self.molecule.GetPointData().GetArray(name))

    def append_bond(self, atom1, atom2, order=1):
        """Add a bond between two atoms.

        Arguments:
            atom1 (Atom or int): The first atom or its ID.
            atom2 (Atom or int): The second atom or its ID.
            order (int): Bond order (1=single, 2=double, 3=triple). Default is 1.
        """
        atom1_id = atom1.atom_id if isinstance(atom1, Atom) else atom1
        atom2_id = atom2.atom_id if isinstance(atom2, Atom) else atom2
        self.molecule.AppendBond(atom1_id, atom2_id, order)

    def get_number_of_atoms(self):
        """Get the number of atoms in the molecule.

        Returns:
            int: Number of atoms.
        """
        return self.molecule.GetNumberOfAtoms()

    def get_number_of_bonds(self):
        """Get the number of bonds in the molecule.

        Returns:
            int: Number of bonds.
        """
        return self.molecule.GetNumberOfBonds()

    def get_bond(self, bond_id):
        """Get bond information by ID (simplified as VTK bond access is limited).

        Arguments:
            bond_id (int): The ID of the bond.

        Returns:
            tuple: (atom1_id, atom2_id, order).
        """
        if bond_id >= self.get_number_of_bonds():
            raise ValueError(f"Bond ID {bond_id} exceeds number of bonds.")
        bond = self.molecule.GetBond(bond_id)
        return (bond.GetBeginAtomId(), bond.GetEndAtomId(), bond.GetOrder())

    def get_atom_positions(self):
        """Get the positions of all atoms.

        Returns:
            np.ndarray: Array of shape (n_atoms, 3) with [x, y, z] coordinates.
        """
        n_atoms = self.get_number_of_atoms()
        positions = np.zeros((n_atoms, 3))
        for i in range(n_atoms):
            pos = self.molecule.GetAtom(i).GetPosition()
            positions[i] = [pos[0], pos[1], pos[2]]
        return positions

    def set_atom_positions(self, positions):
        """Set the positions of all atoms.

        Arguments:
            positions (np.ndarray): Array of shape (n_atoms, 3) with [x, y, z] coordinates.
        """
        n_atoms = self.get_number_of_atoms()
        positions = np.asarray(positions, dtype=float)
        if positions.shape != (n_atoms, 3):
            raise ValueError(
                f"Expected positions shape ({n_atoms}, 3), got {positions.shape}"
            )
        for i in range(n_atoms):
            self.molecule.GetAtom(i).SetPosition(positions[i])

    def get_atomic_numbers(self):
        """Get the atomic numbers of all atoms.

        Returns:
            list: List of atomic numbers.
        """
        return [
            self.molecule.GetAtom(i).GetAtomicNumber()
            for i in range(self.get_number_of_atoms())
        ]

    def set_atomic_numbers(self, atomic_numbers):
        """Set the atomic numbers of all atoms.

        Arguments:
            atomic_numbers (list): List of atomic numbers.
        """
        n_atoms = self.get_number_of_atoms()
        if len(atomic_numbers) != n_atoms:
            raise ValueError(
                f"Expected {n_atoms} atomic numbers, got {len(atomic_numbers)}"
            )
        for i, num in enumerate(atomic_numbers):
            self.molecule.GetAtom(i).SetAtomicNumber(num)

    # Rendering customization methods
    def use_ball_and_stick(self):
        """Set the molecule to use ball-and-stick representation."""
        self.mapper.UseBallAndStickSettings()
        return self

    def use_space_filling(self):
        """Set the molecule to use space-filling (VDW spheres) representation."""
        self.mapper.UseVDWSpheresSettings()
        return self

    def set_atom_radius_scale(self, scale):
        """Set the scale factor for atom radii.

        Arguments:
            scale (float): Scaling factor for atom spheres.
        """
        self.mapper.SetAtomicRadiusScaleFactor(scale)
        return self

    def set_bond_radius(self, radius):
        """Set the radius of bonds.

        Arguments:
            radius (float): Bond radius in world units.
        """
        self.mapper.SetBondRadius(radius)
        return self


class Protein:
    """
    A Vedo-compatible class for protein ribbon visualization, wrapping vtkProteinRibbonFilter.

    This class generates a ribbon representation of protein structures from PDB files,
    vtkMolecule objects, or vtkPolyData, and integrates with Vedo's rendering system.

    Attributes:
        filter (vtkProteinRibbonFilter): The underlying VTK filter for ribbon generation.
        mapper (vtkPolyDataMapper): Maps the filter's output to renderable data.
        actor (vtkActor): The VTK actor for rendering the ribbon.
    """

    def __init__(self, input_data):
        """
        Initialize the ProteinRibbon with input data.

        Arguments:
            input_data (str or vtkMolecule or vtkPolyData):
                - Path to a PDB file (str)
                - A vtkMolecule object
                - A vtkPolyData object

        Raises:
            ValueError: If the input_data type is not supported.
        """

        # Handle different input types
        if isinstance(input_data, str):
            # Read PDB file using vtkPDBReader
            reader = vtkPDBReader()
            reader.SetFileName(input_data)
            reader.Update()
            self.input_data = reader.GetOutput()
        elif isinstance(input_data, vtkMolecule):
            self.input_data = input_data
        elif isinstance(input_data, vtkPolyData):
            self.input_data = input_data
        else:
            raise ValueError(
                "Input must be a PDB file path, vtkMolecule, or vtkPolyData."
            )

        # Create and configure the ribbon filter
        self.filter = vtkProteinRibbonFilter()
        self.filter.SetInputData(self.input_data)
        self.filter.Update()

        # Set up the mapper and actor for rendering
        self.mapper = vtkPolyDataMapper()
        self.mapper.SetInputConnection(self.filter.GetOutputPort())
        self.actor = vtkActor()
        self.actor.SetMapper(self.mapper)
        self.property = self.actor.GetProperty()

        # Set default visual properties
        self.property.SetColor(1.0, 0.8, 0.6)  # Soft peach color
        self.property.SetOpacity(1.0)

    def set_coil_width(self, width):
        """
        Set the width of the coil regions in the ribbon.

        Arguments:
            width (float): The width of the coil regions.

        Returns:
            self: For method chaining.
        """
        self.filter.SetCoilWidth(width)
        self.filter.Update()
        return self

    def set_helix_width(self, width):
        """
        Set the width of the helix regions in the ribbon.

        Arguments:
            width (float): The width of the helix regions.

        Returns:
            self: For method chaining.
        """
        self.filter.SetHelixWidth(width)
        self.filter.Update()
        return self

    def set_sphere_resolution(self, resolution):
        """
        Set the resolution of spheres used in the ribbon representation.

        Arguments:
            resolution (int): The resolution of the spheres.

        Returns:
            self: For method chaining.
        """
        self.filter.SetSphereResolution(resolution)
        self.filter.Update()
        return self
