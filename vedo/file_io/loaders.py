from __future__ import annotations
"""Object loading and conversion utilities."""

import glob
import os
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

import vedo
import vedo.vtkclasses as vtki
from vedo import utils
from vedo.assembly import Assembly
from vedo.image import Image
from vedo.mesh import Mesh
from vedo.pointcloud import Points
from vedo.volume import Volume

from .network import download, file_info, gunzip

__docformat__ = "google"

def load(inputobj: list | str | os.PathLike, unpack=True, force=False) -> Any:
    """
    Load any vedo objects from file or from the web.

    The output will depend on the file extension. See examples below.
    Unzip is made on the fly, if file ends with `.gz`.
    Can load an object directly from a URL address.

    Arguments:
        unpack : (bool)
            unpack MultiBlockData into a flat list of objects.
        force : (bool)
            when downloading a file ignore any previous cached downloads and force a new one.

    Example:
        ```python
        from vedo import dataurl, load, show
        # Return a list of 2 meshes
        g = load([dataurl+'250.vtk', dataurl+'270.vtk'])
        show(g)
        # Return a list of meshes by reading all files in a directory
        # (if directory contains DICOM files then a Volume is returned)
        g = load('mydicomdir/')
        show(g)
        ```
    """
    if isinstance(inputobj, list):
        inputobj = [str(f) for f in inputobj]
    else:
        inputobj = str(inputobj)

    acts = []
    if utils.is_sequence(inputobj):
        flist = inputobj
    elif isinstance(inputobj, str) and inputobj.startswith("https://"):
        flist = [inputobj]
    else:
        flist = utils.humansort(glob.glob(inputobj))

    for fod in flist:

        if fod.startswith("https://"):
            fod = download(fod, force=force, verbose=False)

        if os.path.isfile(fod):  ### it's a file

            if fod.endswith(".gz"):
                fod = gunzip(fod)

            a = _load_file(fod, unpack)
            acts.append(a)

        elif os.path.isdir(fod):  ### it's a directory or DICOM
            flist = utils.humansort(os.listdir(fod))
            if not flist:
                vedo.logger.warning(f"Cannot load empty directory {fod!r}")
                continue
            is_dicom_dir = any(fname.lower().endswith(".dcm") for fname in flist)
            if is_dicom_dir:  ### it's DICOM
                reader = vtki.new("DICOMImageReader")
                reader.SetDirectoryName(fod)
                reader.Update()
                image = reader.GetOutput()
                vol = Volume(image)
                try:
                    vol.metadata["PixelSpacing"] = reader.GetPixelSpacing()
                    vol.metadata["Width"] = reader.GetWidth()
                    vol.metadata["Height"] = reader.GetHeight()
                    vol.metadata["PositionPatient"] = reader.GetImagePositionPatient()
                    vol.metadata["OrientationPatient"] = reader.GetImageOrientationPatient()
                    vol.metadata["BitsAllocated"] = reader.GetBitsAllocated()
                    vol.metadata["PixelRepresentation"] = reader.GetPixelRepresentation()
                    vol.metadata["NumberOfComponents"] = reader.GetNumberOfComponents()
                    vol.metadata["TransferSyntaxUID"] = reader.GetTransferSyntaxUID()
                    vol.metadata["RescaleSlope"] = reader.GetRescaleSlope()
                    vol.metadata["RescaleOffset"] = reader.GetRescaleOffset()
                    vol.metadata["PatientName"] = reader.GetPatientName()
                    vol.metadata["StudyUID"] = reader.GetStudyUID()
                    vol.metadata["StudyID"] = reader.GetStudyID()
                    vol.metadata["GantryAngle"] = reader.GetGantryAngle()
                except Exception as e:
                    vedo.logger.warning(f"Cannot read DICOM metadata: {e}")
                acts.append(vol)

            else:  ### it's a normal directory
                for ifile in flist:
                    full_path = os.path.join(fod, ifile)
                    if not os.path.isfile(full_path):
                        continue
                    a = _load_file(full_path, unpack)
                    acts.append(a)
        else:
            vedo.logger.error(f"in load(), cannot find {fod}")

    if len(acts) == 1:
        if "numpy" in str(type(acts[0])):
            return acts[0]
        if not acts[0]:
            vedo.logger.error(f"in load(), cannot load {inputobj}")
        return acts[0]

    if len(acts) == 0:
        vedo.logger.error(f"in load(), cannot load {inputobj}")
        return None

    else:
        return acts

########################################################################
def _load_file(filename, unpack):
    fl = str(filename).lower()

    ########################################################## other formats:
    if fl.endswith(".xml") or fl.endswith(".xml.gz") or fl.endswith(".xdmf"):
        # Fenics tetrahedral file
        objt = loadDolfin(filename)
    elif fl.endswith(".neutral") or fl.endswith(".neu"):  # neutral tets
        objt = loadNeutral(filename)
    elif fl.endswith(".gmsh"):  # gmesh file
        objt = loadGmesh(filename)
    elif fl.endswith(".pcd"):  # PCL point-cloud format
        objt = loadPCD(filename)
        objt.properties.SetPointSize(2)
    elif fl.endswith(".off"):
        objt = loadOFF(filename)
    elif fl.endswith(".step") or fl.endswith(".stp"):
        objt = loadSTEP(filename)
    elif fl.endswith(".3ds"):  # 3ds format
        objt = load3DS(filename)
    elif fl.endswith(".wrl"):
        importer = vtki.new("VRMLImporter")
        importer.SetFileName(filename)
        importer.Read()
        importer.Update()
        actors = importer.GetRenderer().GetActors()  # vtkActorCollection
        actors.InitTraversal()
        wacts = []
        for i in range(actors.GetNumberOfItems()):
            act = actors.GetNextActor()
            m = Mesh(act.GetMapper().GetInput())
            m.actor = act
            wacts.append(m)
        objt = Assembly(wacts)
    elif fl.endswith(".glb") or fl.endswith(".gltf"):
        importer = vtki.new("GLTFImporter")
        importer.SetFileName(filename)
        importer.Update()
        actors = importer.GetRenderer().GetActors()  # vtkActorCollection
        actors.InitTraversal()
        wacts = []
        for i in range(actors.GetNumberOfItems()):
            act = actors.GetNextActor()
            m = Mesh(act.GetMapper().GetInput())
            m.actor = act
            wacts.append(m)
        objt = Assembly(wacts)

    ######################################################## volumetric:
    elif fl.endswith((".tif", ".tiff", ".slc", ".vti", ".mhd", ".nrrd", ".nii", ".dem")):
        img = loadImageData(filename)
        objt = Volume(img)

    ######################################################### 2D images:
    elif fl.endswith((".jpg", ".jpeg", ".png", ".bmp", ".gif")):
        if fl.endswith(".gif"):
            from PIL import Image as PILImage, ImageSequence

            img = PILImage.open(filename)
            frames = []
            for frame in ImageSequence.Iterator(img):
                a = np.array(frame.convert("RGB").getdata(), dtype=np.uint8)
                a = a.reshape([frame.size[1], frame.size[0], 3])
                frames.append(Image(a))
            return frames
        if fl.endswith(".png"):
            picr = vtki.new("PNGReader")
        elif fl.endswith(".jpg") or fl.endswith(".jpeg"):
            picr = vtki.new("JPEGReader")
        elif fl.endswith(".bmp"):
            picr = vtki.new("BMPReader")

        picr.SetFileName(filename)
        picr.Update()
        objt = Image(picr.GetOutput())

    ######################################################### multiblock:
    elif fl.endswith(".vtm") or fl.endswith(".vtmb"):
        mbread = vtki.new("XMLMultiBlockDataReader")
        mbread.SetFileName(filename)
        mbread.Update()
        mb = mbread.GetOutput()
        if unpack:
            acts = []
            for i in range(mb.GetNumberOfBlocks()):
                b = mb.GetBlock(i)
                if isinstance(
                    b,
                    (
                        vtki.vtkPolyData,
                        vtki.vtkStructuredGrid,
                        vtki.vtkRectilinearGrid,
                    ),
                ):
                    acts.append(Mesh(b))
                elif isinstance(b, vtki.vtkImageData):
                    acts.append(Volume(b))
                elif isinstance(b, vtki.vtkUnstructuredGrid):
                    acts.append(vedo.UnstructuredGrid(b))
            return acts
        return mb

    ######################################################### assembly:
    elif fl.endswith(".npy"):
        data = np.load(filename, allow_pickle=True)
        try:
            # old format with a single object
            meshs = [from_numpy(dd) for dd in data]
        except TypeError:
            data = data.item()
            meshs = []
            for ad in data["objects"][0]["parts"]:
                obb = from_numpy(ad)
                meshs.append(obb)
        return Assembly(meshs)

    ###########################################################
    elif fl.endswith(".geojson"):
        return loadGeoJSON(filename)

    elif fl.endswith(".pvd"):
        return loadPVD(filename)

    ########################################################### polygonal mesh:
    else:
        if fl.endswith(".vtk"):  # read all legacy vtk types
            reader = vtki.new("DataSetReader")
            reader.ReadAllScalarsOn()
            reader.ReadAllVectorsOn()
            reader.ReadAllTensorsOn()
            reader.ReadAllFieldsOn()
            reader.ReadAllNormalsOn()
            reader.ReadAllColorScalarsOn()
        elif fl.endswith(".ply"):
            reader = vtki.new("PLYReader")
        elif fl.endswith(".obj"):
            reader = vtki.new("OBJReader")
            reader.SetGlobalWarningDisplay(0) # suppress warnings issue #980
        elif fl.endswith(".stl"):
            reader = vtki.new("STLReader")
        elif fl.endswith(".byu") or fl.endswith(".g"):
            reader = vtki.new("BYUReader")
        elif fl.endswith(".foam"):  # OpenFoam
            reader = vtki.new("OpenFOAMReader")
        elif fl.endswith(".vtp"):
            reader = vtki.new("XMLPolyDataReader")
        elif fl.endswith(".vts"):
            reader = vtki.new("XMLStructuredGridReader")
        elif fl.endswith(".vtu"):
            reader = vtki.new("XMLUnstructuredGridReader")
        elif fl.endswith(".vtr"):
            reader = vtki.new("XMLRectilinearGridReader")
        elif fl.endswith(".pvtr"):
            reader = vtki.new("XMLPRectilinearGridReader")
        elif fl.endswith("pvtu"):
            reader = vtki.new("XMLPUnstructuredGridReader")
        elif fl.endswith(".txt") or fl.endswith(".xyz") or fl.endswith(".dat"):
            reader = vtki.new("ParticleReader")  # (format is x, y, z, scalar)
        elif fl.endswith(".facet"):
            reader = vtki.new("FacetReader")
        else:
            return None

        reader.SetFileName(filename)
        reader.Update()
        routput = reader.GetOutput()

        if not routput:
            vedo.logger.error(f"unable to load {filename}")
            return None

        if isinstance(routput, vtki.vtkUnstructuredGrid):
            objt = vedo.UnstructuredGrid(routput)

        else:
            objt = Mesh(routput)
            if fl.endswith(".txt") or fl.endswith(".xyz") or fl.endswith(".dat"):
                objt.point_size(4)

    objt.filename = filename
    objt.file_size, objt.created = file_info(filename)
    return objt



def loadStructuredPoints(filename: str | os.PathLike, as_points=True):
    """
    Load and return a `vtkStructuredPoints` object from file.

    If `as_points` is True, return a `Points` object
    instead of a `vtkStructuredPoints`.
    """
    filename = str(filename)
    reader = vtki.new("StructuredPointsReader")
    reader.SetFileName(filename)
    reader.Update()
    if as_points:
        v2p = vtki.new("ImageToPoints")
        v2p.SetInputData(reader.GetOutput())
        v2p.Update()
        pts = Points(v2p.GetOutput())
        return pts
    return reader.GetOutput()

########################################################################
def loadStructuredGrid(filename: str | os.PathLike):
    """Load and return a `vtkStructuredGrid` object from file."""
    filename = str(filename)
    if filename.endswith(".vts"):
        reader = vtki.new("XMLStructuredGridReader")
    else:
        reader = vtki.new("StructuredGridReader")
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


###################################################################
def load3DS(filename: str | os.PathLike) -> Assembly:
    """Load `3DS` file format from file."""
    filename = str(filename)
    renderer = vtki.vtkRenderer()
    renWin = vtki.vtkRenderWindow()
    renWin.AddRenderer(renderer)

    importer = vtki.new("3DSImporter")
    importer.SetFileName(filename)
    importer.ComputeNormalsOn()
    importer.SetRenderWindow(renWin)
    importer.Update()

    actors = renderer.GetActors()  # vtkActorCollection
    acts = []
    for i in range(actors.GetNumberOfItems()):
        a = actors.GetItemAsObject(i)
        acts.append(a)
    del renWin

    wrapped_acts = []
    for a in acts:
        try:
            newa = Mesh(a.GetMapper().GetInput())
            newa.actor = a
            wrapped_acts.append(newa)
            # print("loaded 3DS object", [a])
        except:
            print("ERROR: cannot load 3DS object part", [a])
    return vedo.Assembly(wrapped_acts)

########################################################################
def loadOFF(filename: str | os.PathLike) -> Mesh:
    """Read the OFF file format (polygonal mesh)."""
    filename = str(filename)
    with open(filename, "r", encoding="UTF-8") as f:
        lines = f.readlines()

    vertices = []
    faces = []
    NumberOfVertices = 0
    i = -1
    for text in lines:
        if len(text) == 0:
            continue
        if text == "\n":
            continue
        if "#" in text:
            continue
        if "OFF" in text:
            continue

        ts = text.split()
        n = len(ts)

        if not NumberOfVertices and n > 1:
            NumberOfVertices, NumberOfFaces = int(ts[0]), int(ts[1])
            continue
        i += 1

        if i < NumberOfVertices and n == 3:
            x, y, z = float(ts[0]), float(ts[1]), float(ts[2])
            vertices.append([x, y, z])

        ids = []
        if NumberOfVertices <= i < (NumberOfVertices + NumberOfFaces + 1) and n > 2:
            ids += [int(xx) for xx in ts[1:]]
            faces.append(ids)

    return Mesh(utils.buildPolyData(vertices, faces))

def loadSTEP(filename: str | os.PathLike, deflection=1.0) -> Mesh:
    """
    Reads a 3D STEP file and returns its mesh representation as vertices and triangles.

    Parameters:
    - filename (str): Path to the STEP file.
    - deflection (float): Linear deflection for meshing accuracy (smaller values yield finer meshes).

    Returns:
    - vertices (list of tuples): List of (x, y, z) coordinates of the mesh vertices.
    - triangles (list of tuples): List of (i, j, k) indices representing the triangles.

    Raises:
    - Exception: If the STEP file cannot be read.
    """
    try:
        from OCC.Core.STEPControl import STEPControl_Reader  # type: ignore
        from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh  # type: ignore
        from OCC.Core.TopExp import TopExp_Explorer  # type: ignore
        from OCC.Core.TopoDS import topods  # type: ignore
        from OCC.Core.BRep import BRep_Tool  # type: ignore
        from OCC.Core.TopAbs import TopAbs_FACE  # type: ignore
        from OCC.Core.TopLoc import TopLoc_Location  # type: ignore
    except ImportError:
        raise ImportError(
            "OCC library not found.\n\nPlease install 'pythonocc-core'. "
            "You can install it using the following command:\n"
            "\t\tconda install -c conda-forge pythonocc-core"
        )

    # Initialize the STEP reader
    reader = STEPControl_Reader()
    status = reader.ReadFile(str(filename))
    if status != 1:  # Check if reading was successful (IFSelect_RetDone = 1)
        raise Exception("Error reading STEP file")

    # Transfer the STEP data into a shape
    reader.TransferRoots()
    shape = reader.OneShape()

    # Mesh the shape with the specified deflection
    mesh = BRepMesh_IncrementalMesh(shape, deflection)
    mesh.Perform()

    # Extract vertices and triangles
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    vertices = []
    triangles = []
    vertex_index = 0

    # Iterate over all faces in the shape
    while explorer.More():
        face = topods.Face(explorer.Current())
        location = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, location)

        if triangulation:
            # Extract vertices from the triangulation
            for i in range(1, triangulation.NbNodes() + 1):
                point = triangulation.Node(i).Transformed(location.Transformation())
                vertices.append((point.X(), point.Y(), point.Z()))

            # Extract triangles with adjusted indices
            for i in range(1, triangulation.NbTriangles() + 1):
                triangle = triangulation.Triangle(i)
                n1, n2, n3 = triangle.Get()  # 1-based indices
                triangles.append((
                    n1 + vertex_index - 1,
                    n2 + vertex_index - 1,
                    n3 + vertex_index - 1
                ))

            # Update the vertex index offset for the next face
            vertex_index += triangulation.NbNodes()

        explorer.Next()

    # Create a mesh object
    mesh = Mesh([vertices, triangles])
    return mesh

########################################################################
def loadGeoJSON(filename: str | os.PathLike) -> Mesh:
    """Load GeoJSON files."""
    filename = str(filename)
    jr = vtki.new("GeoJSONReader")
    jr.SetFileName(filename)
    jr.Update()
    return Mesh(jr.GetOutput())

########################################################################
def loadDolfin(filename: str | os.PathLike) -> Mesh | vedo.TetMesh | None:
    """
    Reads a `Fenics/Dolfin` file format (.xml or .xdmf).

    Return a `Mesh` or a `TetMesh` object.
    """
    filename = str(filename)
    try:
        import dolfin
    except ImportError:
        vedo.logger.error("loadDolfin(): dolfin module not found. Install with:")
        vedo.logger.error("  conda create -n fenics -c conda-forge fenics")
        vedo.logger.error("  conda install conda-forge::mshr")
        vedo.logger.error("  conda activate fenics")
        return None

    if filename.lower().endswith(".xdmf"):
        f = dolfin.XDMFFile(filename)
        m = dolfin.Mesh()
        f.read(m)
    else:
        m = dolfin.Mesh(filename)

    cells = m.cells()
    verts = m.coordinates()

    if cells.size and verts.size:
        if len(cells[0]) == 4:  # tetrahedral mesh
            return vedo.TetMesh([verts, cells])
        elif len(cells[0]) == 3:  # triangular mesh
            return Mesh([verts, cells])

    return None


########################################################################
def loadPVD(filename: str | os.PathLike) -> list[Any] | None:
    """Read paraview files."""
    filename = str(filename)
    import xml.etree.ElementTree as et

    tree = et.parse(filename)

    dname = os.path.dirname(filename)
    if not dname:
        dname = "."

    listofobjs = []
    for coll in tree.getroot():
        for dataset in coll:
            fname = dataset.get("file")
            if not fname:
                continue
            ob = load(dname + "/" + fname)
            tm = dataset.get("timestep")
            if tm:
                ob.time = tm
            listofobjs.append(ob)
    if len(listofobjs) == 1:
        return listofobjs[0]
    if len(listofobjs) == 0:
        return None
    return listofobjs

########################################################################
def loadNeutral(filename: str | os.PathLike) -> vedo.TetMesh:
    """
    Reads a `Neutral` tetrahedral file format.

    Returns an `TetMesh` object.
    """
    filename = str(filename)
    with open(filename, "r", encoding="UTF-8") as f:
        lines = f.readlines()

    ncoords = int(lines[0])
    coords = []
    for i in range(1, ncoords + 1):
        x, y, z = lines[i].split()
        coords.append([float(x), float(y), float(z)])

    ntets = int(lines[ncoords + 1])
    idolf_tets = []
    for i in range(ncoords + 2, ncoords + ntets + 2):
        text = lines[i].split()
        v0, v1, v2, v3 = int(text[1])-1, int(text[2])-1, int(text[3])-1, int(text[4])-1
        idolf_tets.append([v0, v1, v2, v3])

    return vedo.TetMesh([coords, idolf_tets])

########################################################################
def loadGmesh(filename: str | os.PathLike) -> Mesh:
    """Reads a `gmesh` file format. Return an `Mesh` object."""
    filename = str(filename)
    with open(filename, "r", encoding="UTF-8") as f:
        lines = f.readlines()

    nnodes = 0
    index_nodes = 0
    for i, line in enumerate(lines):
        if "$Nodes" in line:
            index_nodes = i + 1
            nnodes = int(lines[index_nodes])
            break
    node_coords = []
    for i in range(index_nodes + 1, index_nodes + 1 + nnodes):
        cn = lines[i].split()
        node_coords.append([float(cn[1]), float(cn[2]), float(cn[3])])

    nelements = 0
    index_elements = 0
    for i, line in enumerate(lines):
        if "$Elements" in line:
            index_elements = i + 1
            nelements = int(lines[index_elements])
            break
    elements = []
    for i in range(index_elements + 1, index_elements + 1 + nelements):
        ele = lines[i].split()
        elements.append([int(ele[-3]), int(ele[-2]), int(ele[-1])])

    poly = utils.buildPolyData(node_coords, elements, index_offset=1)
    return Mesh(poly)

########################################################################
def loadPCD(filename: str | os.PathLike) -> Points:
    """Return a `Mesh` made of only vertex points
    from the `PointCloud` library file format.

    Returns an `Points` object.
    """
    filename = str(filename)
    with open(filename, "r", encoding="UTF-8") as f:
        lines = f.readlines()

    start = False
    pts = []
    N, expN = 0, 0
    for text in lines:
        if start:
            if N >= expN:
                break
            l = text.split()
            pts.append([float(l[0]), float(l[1]), float(l[2])])
            N += 1
        if not start and "POINTS" in text:
            expN = int(text.split()[1])
        if not start and "DATA ascii" in text:
            start = True
    if expN != N:
        vedo.logger.warning(f"Mismatch in PCD file {expN} != {len(pts)}")
    poly = utils.buildPolyData(pts)
    return Points(poly).point_size(4)

#########################################################################
def from_numpy(d: dict) -> Mesh:
    """Create a Mesh object from a dictionary."""
    # recreate a mesh from numpy arrays
    keys = d.keys()

    points = d["points"]
    cells = d["cells"] if "cells" in keys else None
    lines = d["lines"] if "lines" in keys else None

    msh = Mesh([points, cells, lines])

    if "pointdata" in keys and isinstance(d["pointdata"], dict):
        for arrname, arr in d["pointdata"].items():
            msh.pointdata[arrname] = arr
    if "celldata" in keys and isinstance(d["celldata"], dict):
        for arrname, arr in d["celldata"].items():
            msh.celldata[arrname] = arr
    if "metadata" in keys and isinstance(d["metadata"], dict):
        for arrname, arr in d["metadata"].items():
            msh.metadata[arrname] = arr

    prp = msh.properties
    prp.SetAmbient(d['ambient'])
    prp.SetDiffuse(d['diffuse'])
    prp.SetSpecular(d['specular'])
    prp.SetSpecularPower(d['specularpower'])
    prp.SetSpecularColor(d['specularcolor'])

    prp.SetInterpolation(0)
    # prp.SetInterpolation(d['shading'])

    prp.SetOpacity(d['alpha'])
    prp.SetRepresentation(d['representation'])
    prp.SetPointSize(d['pointsize'])
    if d['color'] is not None:
        msh.color(d['color'])
    if "lighting_is_on" in d.keys():
        prp.SetLighting(d['lighting_is_on'])
    # Must check keys for backwards compatibility:
    if "linecolor" in d.keys() and d['linecolor'] is not None:
        msh.linecolor(d['linecolor'])
    if "backcolor" in d.keys() and d['backcolor'] is not None:
        msh.backcolor(d['backcolor'])

    if d['linewidth'] is not None:
        msh.linewidth(d['linewidth'])
    if "edge_visibility" in d.keys():
        prp.SetEdgeVisibility(d['edge_visibility']) # new

    lut_list  = d["LUT"]
    lut_range = d["LUT_range"]
    ncols = len(lut_list)
    lut = vtki.vtkLookupTable()
    lut.SetNumberOfTableValues(ncols)
    lut.SetRange(lut_range)
    for i in range(ncols):
        r, g, b, a = lut_list[i]
        lut.SetTableValue(i, r, g, b, a)
    lut.Build()
    msh.mapper.SetLookupTable(lut)
    msh.mapper.SetScalarRange(lut_range)

    try: # NEW in vedo 5.0
        arname = d["array_name_to_color_by"]
        msh.mapper.SetArrayName(arname)
        msh.mapper.SetInterpolateScalarsBeforeMapping(
            d["interpolate_scalars_before_mapping"])
        msh.mapper.SetUseLookupTableScalarRange(
            d["use_lookup_table_scalar_range"])
        msh.mapper.SetScalarRange(d["scalar_range"])
        msh.mapper.SetScalarVisibility(d["scalar_visibility"])
        msh.mapper.SetScalarMode(d["scalar_mode"])
        msh.mapper.SetColorMode(d["color_mode"])
        if d["scalar_visibility"]:
            if d["scalar_mode"] == 1:
                msh.dataset.GetPointData().SetActiveScalars(arname)
            if d["scalar_mode"] == 2:
                msh.dataset.GetCellData().SetActiveScalars(arname)

        if "texture_array" in keys and d["texture_array"] is not None:
            # recreate a vtkTexture object from numpy arrays:
            t = vtki.vtkTexture()
            t.SetInterpolate(d["texture_interpolate"])
            t.SetRepeat(d["texture_repeat"])
            t.SetQuality(d["texture_quality"])
            t.SetColorMode(d["texture_color_mode"])
            t.SetMipmap(d["texture_mipmap"])
            t.SetBlendingMode(d["texture_blending_mode"])
            t.SetEdgeClamp(d["texture_edge_clamp"])
            t.SetBorderColor(d["texture_border_color"])
            msh.actor.SetTexture(t)
            tcarray = None
            for arrname in msh.pointdata.keys():
                if "Texture" in arrname or "TCoord" in arrname:
                    tcarray = arrname
                    break
            if tcarray is not None:
                t.SetInputData(vedo.Image(d["texture_array"]).dataset)
                msh.pointdata.select_texture_coords(tcarray)

        # print("color_mode", d["color_mode"])
        # print("scalar_mode", d["scalar_mode"])
        # print("scalar_range", d["scalar_range"])
        # print("scalar_visibility", d["scalar_visibility"])
        # print("array_name_to_color_by", arname)
    except KeyError:
        pass

    if "time" in keys: msh.time = d["time"]
    if "name" in keys: msh.name = d["name"]
    # if "info" in keys: msh.info = d["info"]
    if "filename" in keys: msh.filename = d["filename"]
    if "pickable" in keys: msh.pickable(d["pickable"])
    if "dragable" in keys: msh.draggable(d["dragable"])
    return msh

#############################################################################
def _import_npy(fileinput: str | os.PathLike) -> vedo.Plotter:
    """Import a vedo scene from numpy format."""
    fileinput = str(fileinput)

    fileinput = download(fileinput, verbose=False, force=True)
    if fileinput.endswith(".npy"):
        data = np.load(fileinput, allow_pickle=True, encoding="latin1").flatten()[0]
    elif fileinput.endswith(".npz"):
        data = np.load(fileinput, allow_pickle=True)["vedo_scenes"][0]

    if "use_parallel_projection" in data.keys():
        vedo.settings.use_parallel_projection = data["use_parallel_projection"]
    if "use_polygon_offset" in data.keys():
        vedo.settings.use_polygon_offset = data["use_polygon_offset"]
    if "polygon_offset_factor" in data.keys():
        vedo.settings.polygon_offset_factor = data["polygon_offset_factor"]
    if "polygon_offset_units" in data.keys():
        vedo.settings.polygon_offset_units = data["polygon_offset_units"]
    if "interpolate_scalars_before_mapping" in data.keys():
        vedo.settings.interpolate_scalars_before_mapping = data["interpolate_scalars_before_mapping"]
    if "default_font" in data.keys():
        vedo.settings.default_font = data["default_font"]
    if "use_depth_peeling" in data.keys():
        vedo.settings.use_depth_peeling = data["use_depth_peeling"]

    axes = data.pop("axes", 4) # UNUSED
    title = data.pop("title", "")
    backgrcol  = data.pop("backgrcol", "white")
    backgrcol2 = data.pop("backgrcol2", None)
    cam = data.pop("camera", None)

    if data["shape"] != (1, 1):
        data["size"] = "auto"  # disable size

    plt = vedo.Plotter(
        size=data["size"],  # not necessarily a good idea to set it
        axes=axes,          # must be zero to avoid recreating the axes
        title=title,
        bg=backgrcol,
        bg2=backgrcol2,
    )

    if cam:
        if "pos" in cam.keys():
            plt.camera.SetPosition(cam["pos"])
        if "focalPoint" in cam.keys(): # obsolete
            plt.camera.SetFocalPoint(cam["focalPoint"])
        if "focal_point" in cam.keys():
            plt.camera.SetFocalPoint(cam["focal_point"])
        if "viewup" in cam.keys():
            plt.camera.SetViewUp(cam["viewup"])
        if "distance" in cam.keys():
            plt.camera.SetDistance(cam["distance"])
        if "clippingRange" in cam.keys(): # obsolete
            plt.camera.SetClippingRange(cam["clippingRange"])
        if "clipping_range" in cam.keys():
            plt.camera.SetClippingRange(cam["clipping_range"])
        if "parallel_scale" in cam.keys():
            plt.camera.SetParallelScale(cam["parallel_scale"])

    ##############################################
    objs = []
    for d in data["objects"]:
        ### Mesh
        if d['type'].lower() == 'mesh':
            obj = from_numpy(d)

        ### Assembly
        elif d['type'].lower() == 'assembly':
            assacts = []
            for ad in d["actors"]:
                assacts.append(from_numpy(ad))
            obj = Assembly(assacts)
            obj.SetScale(d["scale"])
            obj.SetPosition(d["position"])
            obj.SetOrientation(d["orientation"])
            obj.SetOrigin(d["origin"])

        ### Volume
        elif d['type'].lower() == 'volume':
            obj = Volume(d["array"])
            obj.spacing(d["spacing"])
            obj.origin(d["origin"])
            if "jittering" in d.keys(): obj.jittering(d["jittering"])
            obj.mode(d["mode"])
            obj.color(d["color"])
            obj.alpha(d["alpha"])
            obj.alpha_gradient(d["alphagrad"])

        ### TetMesh
        elif d['type'].lower() == 'tetmesh':
            raise NotImplementedError("TetMesh not supported yet")

        ### ScalarBar2D
        elif d['type'].lower() == 'scalarbar2d':
            raise NotImplementedError("ScalarBar2D not supported yet")

        ### Image
        elif d['type'].lower() == 'image':
            obj = Image(d["array"])
            obj.alpha(d["alpha"])
            obj.actor.SetScale(d["scale"])
            obj.actor.SetPosition(d["position"])
            obj.actor.SetOrientation(d["orientation"])
            obj.actor.SetOrigin(d["origin"])

        ### Text2D
        elif d['type'].lower() == 'text2d':
            obj = vedo.shapes.Text2D(d["text"], font=d["font"], c=d["color"])
            obj.pos(d["position"]).size(d["size"])
            obj.background(d["bgcol"], d["alpha"])
            if d["frame"]:
                obj.frame(d["bgcol"])

        else:
            obj = None
            # vedo.logger.warning(f"Cannot import object {d}")

        if obj:
            keys = d.keys()
            if "time" in keys: obj.time = d["time"]
            if "name" in keys: obj.name = d["name"]
            # if "info" in keys: obj.info = d["info"]
            if "filename" in keys: obj.filename = d["filename"]
            objs.append(obj)

    plt.add(objs)
    plt.resetcam = False
    return plt

###########################################################
def loadImageData(filename: str | os.PathLike) -> vtki.vtkImageData | None:
    """Read and return a `vtkImageData` object from file."""
    filename = str(filename)
    if ".ome.tif" in filename.lower():
        reader = vtki.new("OMETIFFReader")
        # print("GetOrientationType ", reader.GetOrientationType())
        reader.SetOrientationType(vedo.settings.tiff_orientation_type)
    elif ".tif" in filename.lower():
        reader = vtki.new("TIFFReader")
        # print("GetOrientationType ", reader.GetOrientationType())
        reader.SetOrientationType(vedo.settings.tiff_orientation_type)
    elif ".slc" in filename.lower():
        reader = vtki.new("SLCReader")
        if not reader.CanReadFile(filename):
            vedo.logger.error(f"sorry, bad SLC file {filename}")
            return None
    elif ".vti" in filename.lower():
        reader = vtki.new("XMLImageDataReader")
    elif ".mhd" in filename.lower():
        reader = vtki.new("MetaImageReader")
    elif ".dem" in filename.lower():
        reader = vtki.new("DEMReader")
    elif ".nii" in filename.lower():
        reader = vtki.new("NIFTIImageReader")
    elif ".nrrd" in filename.lower():
        reader = vtki.new("NrrdReader")
        if not reader.CanReadFile(filename):
            vedo.logger.error(f"sorry, bad NRRD file {filename}")
            return None
    else:
        vedo.logger.error(f"cannot read file {filename}")
        return None
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()

###########################################################

def load_obj(fileinput: str | os.PathLike, mtl_file=None, texture_path=None) -> list[Mesh]:
    """
    Import a set of meshes from a OBJ wavefront file.

    Arguments:
        mtl_file : (str)
            MTL file for OBJ wavefront files
        texture_path : (str)
            path of the texture files directory

    Returns:
        `list(Mesh)`
    """
    fileinput = str(fileinput)

    window = vtki.vtkRenderWindow()
    window.SetOffScreenRendering(1)
    renderer = vtki.vtkRenderer()
    window.AddRenderer(renderer)

    importer = vtki.new("OBJImporter")
    importer.SetFileName(fileinput)
    if mtl_file is None:
        mtl_file = fileinput.replace(".obj", ".mtl").replace(".OBJ", ".MTL")
    if os.path.isfile(mtl_file):
        importer.SetFileNameMTL(mtl_file)
    if texture_path is None:
        texture_path = fileinput.replace(".obj", ".txt").replace(".OBJ", ".TXT")
    # since the texture_path may be a directory which contains textures
    if os.path.exists(texture_path):
        importer.SetTexturePath(texture_path)
    importer.SetRenderWindow(window)
    importer.Update()

    actors = renderer.GetActors()
    actors.InitTraversal()
    objs = []
    for _ in range(actors.GetNumberOfItems()):
        vactor = actors.GetNextActor()
        msh = Mesh(vactor)
        msh.name = "OBJMesh"
        msh.copy_properties_from(vactor)
        tx = vactor.GetTexture()
        if tx:
            msh.texture(tx)
        objs.append(msh)
    return objs

__all__ = [
    "load",
    "_load_file",
    "loadStructuredPoints",
    "loadStructuredGrid",
    "load3DS",
    "loadOFF",
    "loadSTEP",
    "loadGeoJSON",
    "loadDolfin",
    "loadPVD",
    "loadNeutral",
    "loadGmesh",
    "loadPCD",
    "from_numpy",
    "_import_npy",
    "loadImageData",
    "load_obj",
]
