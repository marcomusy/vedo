from __future__ import division, print_function
import vtk
import os
import sys
import time
import numpy

import vtkplotter.utils as utils
import vtkplotter.colors as colors
from vtkplotter.actors import Actor, Assembly, ImageActor, isosurface
import vtkplotter.docs as docs
import vtkplotter.settings as settings
import vtkplotter.addons as addons

__doc__ = (
    """
Submodule to load meshes of different formats, and other I/O functionalities.
"""
    + docs._defs
)

__all__ = [
    "load",
    "loadPolyData",
    "loadXMLGenericData",
    "loadStructuredPoints",
    "loadStructuredGrid",
    "loadUnStructuredGrid",
    "loadRectilinearGrid",
    "loadMultiBlockData",
    "load3DS",
    "loadDolfin",
    "loadNeutral",
    "loadGmesh",
    "loadPCD",
    "loadOFF",
    "loadDICOM",
    "loadImageData",
    "load2Dimage",
    "write",
    "screenshot",
    "Video",
    "ProgressBar",
    "convertNeutral2Xml",
    "buildPolyData",
    "Button",
]


def load(
    inputobj,
    c="gold",
    alpha=None,
    wire=False,
    bc=None,
    texture=None,
    smoothing=None,
    threshold=None,
    connectivity=False,
):
    """
    Returns a ``vtkActor`` from reading a file, directory or ``vtkPolyData``.

    :param c: color in RGB format, hex, symbol or name
    :param alpha:   transparency (0=invisible)
    :param wire:    show surface as wireframe
    :param bc:      backface color of internal surface
    :param texture: any png/jpg file can be used as texture

    For volumetric data (tiff, slc, vti files):

    :param smoothing:    gaussian filter to smooth vtkImageData
    :param threshold:    value to draw the isosurface
    :param connectivity: if True only keeps the largest portion of the polydata
    """
    if alpha is None:
        alpha = 1

    if isinstance(inputobj, vtk.vtkPolyData):
        a = Actor(inputobj, c, alpha, wire, bc, texture)
        if inputobj and inputobj.GetNumberOfPoints() == 0:
            colors.printc("~lightning Warning: actor has zero points.", c=5)
        return a

    acts = []
    if isinstance(inputobj, list):
        flist = inputobj
    else:
        import glob
        flist = sorted(glob.glob(inputobj))

    for fod in flist:
        if os.path.isfile(fod):
            if fod.endswith(".vtm"):
                acts += loadMultiBlockData(fod, unpack=True)
            else:
                a = _loadFile(fod, c, alpha, wire, bc, texture,
                              smoothing, threshold, connectivity)
                acts.append(a)
        elif os.path.isdir(fod):
            acts = _loadDir(fod, c, alpha, wire, bc, texture,
                            smoothing, threshold, connectivity)
    if not len(acts):
        colors.printc("~times Error in load(): cannot find", inputobj, c=1)
        return None

    if len(acts) == 1:
        return acts[0]
    else:
        return acts


def _loadFile(filename, c, alpha, wire, bc, texture, smoothing, threshold, connectivity):
    fl = filename.lower()
    if fl.endswith(".xml") or fl.endswith(".xml.gz"):  # Fenics tetrahedral file
        actor = loadDolfin(filename, c, alpha, wire, bc)
    elif fl.endswith(".neutral") or fl.endswith(".neu"):  # neutral tetrahedral file
        actor = loadNeutral(filename, c, alpha, wire, bc)
    elif fl.endswith(".gmsh"):  # gmesh file
        actor = loadGmesh(filename, c, alpha, wire, bc)
    elif fl.endswith(".pcd"):  # PCL point-cloud format
        actor = loadPCD(filename, c, alpha)
    elif fl.endswith(".off"):
        actor = loadOFF(filename, c, alpha, wire, bc)
    elif fl.endswith(".3ds"):  # 3ds point-cloud format
        actor = load3DS(filename)
    elif fl.endswith(".tif") or fl.endswith(".slc") or fl.endswith(".vti") or fl.endswith(".mhd"):
        # tiff stack or slc mhd, or vti
        img = loadImageData(filename)
        actor = isosurface(img, smoothing, threshold, connectivity)
    elif fl.endswith(".png") or fl.endswith(".jpg") or fl.endswith(".bmp") or fl.endswith(".jpeg"):
        actor = load2Dimage(filename, alpha)
    else:
        poly = loadPolyData(filename)
        if not poly:
            colors.printc("~noentry Unable to load", filename, c=1)
            return None
        actor = Actor(poly, c, alpha, wire, bc, texture)
        if fl.endswith(".txt") or fl.endswith(".xyz"):
            actor.GetProperty().SetPointSize(4)
    actor.filename = filename
    return actor


def _loadDir(mydir, c, alpha, wire, bc, texture, smoothing, threshold, connectivity):
    if not os.path.exists(mydir):
        colors.printc("~noentry Error in loadDir: Cannot find", mydir, c=1)
        exit(0)
    acts = []
    flist = os.listdir(mydir)
    utils.humansort(flist)
    for ifile in flist:
        a = _loadFile(mydir+'/'+ifile, c, alpha, wire, bc, texture,
                      smoothing, threshold, connectivity)
        acts.append(a)
    return acts


def loadPolyData(filename):
    """Load a file and return a ``vtkPolyData`` object (not a ``vtkActor``)."""
    if not os.path.exists(filename):
        colors.printc("~noentry Error in loadPolyData: Cannot find", filename, c=1)
        return None
    fl = filename.lower()
    if   fl.endswith(".vtk"):
        reader = vtk.vtkPolyDataReader()
    elif fl.endswith(".ply"):
        reader = vtk.vtkPLYReader()
    elif fl.endswith(".obj"):
        reader = vtk.vtkOBJReader()
    elif fl.endswith(".stl"):
        reader = vtk.vtkSTLReader()
    elif fl.endswith(".byu") or fl.endswith(".g"):
        reader = vtk.vtkBYUReader()
    elif fl.endswith(".vtp"):
        reader = vtk.vtkXMLPolyDataReader()
    elif fl.endswith(".vts"):
        reader = vtk.vtkXMLStructuredGridReader()
    elif fl.endswith(".vtu"):
        reader = vtk.vtkXMLUnstructuredGridReader()
    elif fl.endswith(".txt"):
        reader = vtk.vtkParticleReader()  # (x y z scalar)
    elif fl.endswith(".xyz"):
        reader = vtk.vtkParticleReader()
    else:
        reader = vtk.vtkDataReader()
    reader.SetFileName(filename)
    if fl.endswith(".vts"):  # structured grid
        reader.Update()
        gf = vtk.vtkStructuredGridGeometryFilter()
        gf.SetInputConnection(reader.GetOutputPort())
        gf.Update()
        poly = gf.GetOutput()
    elif fl.endswith(".vtu"):  # unstructured grid
        reader.Update()
        gf = vtk.vtkGeometryFilter()
        gf.SetInputConnection(reader.GetOutputPort())
        gf.Update()
        poly = gf.GetOutput()
    else:
        try:
            reader.Update()
            poly = reader.GetOutput()
        except:
            poly = None

    if not poly:
        return None

    cleanpd = vtk.vtkCleanPolyData()
    cleanpd.SetInputData(poly)
    cleanpd.Update()
    return cleanpd.GetOutput()


def loadMultiBlockData(filename, unpack=True):
    read = vtk.vtkXMLMultiBlockDataReader()
    read.SetFileName(filename)
    read.Update()
    mb = read.GetOutput()
    if unpack:
        acts = []
        for i in range(mb.GetNumberOfBlocks()):
            b =  mb.GetBlock(i)
            if isinstance(b, (vtk.vtkPolyData,
                              vtk.vtkImageData,
                              vtk.vtkUnstructuredGrid,
                              vtk.vtkStructuredGrid,
                              vtk.vtkRectilinearGrid)):
                acts.append(b)
        return acts
    else:
        return mb
        


def loadXMLGenericData(filename):  # not tested
    """Read any type of vtk data object encoded in XML format. Return an ``Actor(vtkActor)`` object."""
    reader = vtk.vtkXMLGenericDataObjectReader()
    reader.SetFileName(filename)
    reader.Update()
    return Actor(reader.GetOutput())


def loadStructuredPoints(filename):
    """Load a ``vtkStructuredPoints`` object from file and return an ``Actor(vtkActor)`` object.

    .. hint:: |readStructuredPoints| |readStructuredPoints.py|_
    """
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.Update()
    gf = vtk.vtkImageDataGeometryFilter()
    gf.SetInputConnection(reader.GetOutputPort())
    gf.Update()
    return Actor(gf.GetOutput())


def loadStructuredGrid(filename):  # not tested
    """Load a ``vtkStructuredGrid`` object from file and return a ``Actor(vtkActor)`` object."""
    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    gf = vtk.vtkStructuredGridGeometryFilter()
    gf.SetInputConnection(reader.GetOutputPort())
    gf.Update()
    return Actor(gf.GetOutput())


def loadUnStructuredGrid(filename):  # not tested
    """Load a ``vtkunStructuredGrid`` object from file and return a ``Actor(vtkActor)`` object."""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    gf = vtk.vtkUnstructuredGridGeometryFilter()
    gf.SetInputConnection(reader.GetOutputPort())
    gf.Update()
    return Actor(gf.GetOutput())


def loadRectilinearGrid(filename):  # not tested
    """Load a ``vtkRectilinearGrid`` object from file and return a ``Actor(vtkActor)`` object."""
    reader = vtk.vtkRectilinearGridReader()
    reader.SetFileName(filename)
    reader.Update()
    gf = vtk.vtkRectilinearGridGeometryFilter()
    gf.SetInputConnection(reader.GetOutputPort())
    gf.Update()
    return Actor(gf.GetOutput())


def load3DS(filename):
    """Load ``3DS`` file format from file. Return an ``Assembly(vtkAssembly)`` object."""
    renderer = vtk.vtkRenderer()
    renWin = vtk.vtkRenderWindow()
    renWin.AddRenderer(renderer)

    importer = vtk.vtk3DSImporter()
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
    return Assembly(acts)


def loadOFF(filename, c="gold", alpha=1, wire=False, bc=None):
    """Read OFF file format."""
    if not os.path.exists(filename):
        colors.printc("~noentry Error in loadOFF: Cannot find", filename, c=1)
        return None

    f = open(filename, "r")
    lines = f.readlines()
    f.close()

    vertices = []
    faces = []
    NumberOfVertices = None
    i = -1
    for text in lines:
        if len(text) == 0:
            continue
        if text == '\n':
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
            ids += [int(x) for x in ts[1:]]
            faces.append(ids)

    return Actor(buildPolyData(vertices, faces), c, alpha, wire, bc)


def loadDolfin(filename, c="gold", alpha=0.5, wire=None, bc=None):
    """Reads a `Fenics/Dolfin` file format. Return an ``Actor(vtkActor)`` object."""
    if not os.path.exists(filename):
        colors.printc("~noentry Error in loadDolfin: Cannot find", filename, c=1)
        return None
    import xml.etree.ElementTree as et

    if filename.endswith(".gz"):
        import gzip

        inF = gzip.open(filename, "rb")
        outF = open("/tmp/filename.xml", "wb")
        outF.write(inF.read())
        outF.close()
        inF.close()
        tree = et.parse("/tmp/filename.xml")
    else:
        tree = et.parse(filename)
    coords, connectivity = [], []
    for mesh in tree.getroot():
        for elem in mesh:
            for e in elem.findall("vertex"):
                x = float(e.get("x"))
                y = float(e.get("y"))
                ez = e.get("z")
                if ez is None:
                    coords.append([x, y])
                else:
                    z = float(ez)
                    coords.append([x, y, z])

            tets = elem.findall("tetrahedron")
            if not len(tets):
                tris = elem.findall("triangle")
                for e in tris:
                    v0 = int(e.get("v0"))
                    v1 = int(e.get("v1"))
                    v2 = int(e.get("v2"))
                    connectivity.append([v0, v1, v2])
            else:
                for e in tets:
                    v0 = int(e.get("v0"))
                    v1 = int(e.get("v1"))
                    v2 = int(e.get("v2"))
                    v3 = int(e.get("v3"))
                    connectivity.append([v0, v1, v2, v3])

    poly = buildPolyData(coords, connectivity)
    return Actor(poly, c, alpha, True, bc)


def loadNeutral(filename, c="gold", alpha=1, wire=False, bc=None):
    """Reads a `Neutral` tetrahedral file format. Return an ``Actor(vtkActor)`` object."""
    if not os.path.exists(filename):
        colors.printc("~noentry Error in loadNeutral: Cannot find", filename, c=1)
        return None

    coords, connectivity = convertNeutral2Xml(filename)
    poly = buildPolyData(coords, connectivity, indexOffset=0)
    return Actor(poly, c, alpha, wire, bc)


def loadGmesh(filename, c="gold", alpha=1, wire=False, bc=None):
    """Reads a `gmesh` file format. Return an ``Actor(vtkActor)`` object."""
    if not os.path.exists(filename):
        colors.printc("~noentry Error in loadGmesh: Cannot find", filename, c=1)
        return None

    f = open(filename, "r")
    lines = f.readlines()
    f.close()

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

    poly = buildPolyData(node_coords, elements, indexOffset=1)

    return Actor(poly, c, alpha, wire, bc)


def loadPCD(filename, c="gold", alpha=1):
    """Return ``vtkActor`` from `Point Cloud` file format. Return an ``Actor(vtkActor)`` object."""
    if not os.path.exists(filename):
        colors.printc("~noentry Error in loadPCD: Cannot find file", filename, c=1)
        return None
    f = open(filename, "r")
    lines = f.readlines()
    f.close()
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
        colors.printc("~!? Mismatch in pcd file", expN, len(pts), c="red")
    src = vtk.vtkPointSource()
    src.SetNumberOfPoints(len(pts))
    src.Update()
    poly = src.GetOutput()
    for i, p in enumerate(pts):
        poly.GetPoints().SetPoint(i, p)
    if not poly:
        colors.printc("~noentry Unable to load", filename, c="red")
        return False
    actor = Actor(poly, colors.getColor(c), alpha)
    actor.GetProperty().SetPointSize(4)
    return actor


def loadDICOM(dirname, spacing=()):
    reader = vtk.vtkDICOMImageReader()
    reader.SetDirectoryName(dirname)
    reader.Update()
    image = reader.GetOutput()
    print("scalar range:", image.GetScalarRange())
    #colors.printHistogram()
    if len(spacing) == 3:
        image.SetSpacing(spacing[0], spacing[1], spacing[2])
    return image


def loadImageData(filename, spacing=()):
    """Read and return a ``vtkImageData`` object from file."""
    if not os.path.isfile(filename):
        colors.printc("~noentry File not found:", filename, c=1)
        return None

    if ".tif" in filename.lower():
        reader = vtk.vtkTIFFReader()
    elif ".slc" in filename.lower():
        reader = vtk.vtkSLCReader()
        if not reader.CanReadFile(filename):
            colors.printc("~prohibited Sorry bad slc file " + filename, c=1)
            exit(1)
    elif ".vti" in filename.lower():
        reader = vtk.vtkXMLImageDataReader()
    elif ".mhd" in filename.lower():
        reader = vtk.vtkMetaImageReader()
    reader.SetFileName(filename)
    reader.Update()
    image = reader.GetOutput()
    if len(spacing) == 3:
        image.SetSpacing(spacing[0], spacing[1], spacing[2])
    return image


###########################################################
def load2Dimage(filename, alpha=1):
    """Read a JPEG/PNG/BMP image from file. Return an ``ImageActor(vtkImageActor)`` object.

    .. hint:: |rotateImage| |rotateImage.py|_
    """
    fl = filename.lower()
    if ".png" in fl:
        picr = vtk.vtkPNGReader()
    elif ".jpg" in fl or ".jpeg" in fl:
        picr = vtk.vtkJPEGReader()
    elif ".bmp" in fl:
        picr = vtk.vtkBMPReader()
        picr.Allow8BitBMPOff()
    else:
        colors.printc("~times File must end with png, bmp or jp(e)g", c=1)
        exit(1)
    picr.SetFileName(filename)
    picr.Update()
    vactor = ImageActor()  # vtk.vtkImageActor()
    vactor.SetInputData(picr.GetOutput())
    if alpha is None:
        alpha = 1
    vactor.SetOpacity(alpha)
    return vactor


def write(objct, fileoutput, binary=True):
    """
    Write 3D object to file.

    Possile extensions are:
        - vtk, vti, ply, obj, stl, byu, vtp, xyz, tif, png, bmp.
    """
    obj = objct
    if isinstance(obj, Actor):
        obj = objct.polydata(True)
    elif isinstance(obj, vtk.vtkActor):
        obj = objct.GetMapper().GetInput()

    fr = fileoutput.lower()
    if ".vtk" in fr:
        w = vtk.vtkPolyDataWriter()
    elif ".ply" in fr:
        w = vtk.vtkPLYWriter()
    elif ".stl" in fr:
        w = vtk.vtkSTLWriter()
    elif ".vtp" in fr:
        w = vtk.vtkXMLPolyDataWriter()
    elif ".vtm" in fr:
        g = vtk.vtkMultiBlockDataGroupFilter()
        for ob in objct:
            g.AddInputData(ob)
        g.Update()
        mb = g.GetOutputDataObject(0)
        wri = vtk.vtkXMLMultiBlockDataWriter()
        wri.SetInputData(mb)
        wri.SetFileName(fileoutput)
        wri.Write()
        return mb
    elif ".xyz" in fr:
        w = vtk.vtkSimplePointsWriter()
    elif ".byu" in fr or fr.endswith(".g"):
        w = vtk.vtkBYUWriter()
    elif ".obj" in fr:
        w = vtk.vtkOBJExporter()
        w.SetFilePrefix(fileoutput.replace(".obj", ""))
        colors.printc("~target Please use write(vp.window)", c=3)
        w.SetInputData(obj)
        w.Update()
        colors.printc("~save Saved file: " + fileoutput, c="g")
        return objct
    elif ".tif" in fr:
        w = vtk.vtkTIFFWriter()
        w.SetFileDimensionality(len(obj.GetDimensions()))
    elif ".vti" in fr:
        w = vtk.vtkXMLImageDataWriter()
    elif ".png" in fr:
        w = vtk.vtkPNGWriter()
    elif ".jpg" in fr:
        w = vtk.vtkJPEGWriter()
    elif ".bmp" in fr:
        w = vtk.vtkBMPWriter()
    else:
        colors.printc("~noentry Unknown format", fileoutput, "file not saved.", c="r")
        return objct

    try:
        if not ".tif" in fr and not ".vti" in fr:
            if binary and not ".tif" in fr and not ".vti" in fr:
                w.SetFileTypeToBinary()
            else:
                w.SetFileTypeToASCII()
        w.SetInputData(obj)
        w.SetFileName(fileoutput)
        w.Write()
        colors.printc("~save Saved file: " + fileoutput, c="g")
    except Exception as e:
        colors.printc("~noentry Error saving: " + fileoutput, "\n", e, c="r")
    return objct


def convertNeutral2Xml(infile, outfile=None):
    """Convert Neutral file format to Dolfin XML."""

    f = open(infile, "r")
    lines = f.readlines()
    f.close()

    ncoords = int(lines[0])
    fdolf_coords = []
    for i in range(1, ncoords + 1):
        x, y, z = lines[i].split()
        fdolf_coords.append([float(x), float(y), float(z)])

    ntets = int(lines[ncoords + 1])
    idolf_tets = []
    for i in range(ncoords + 2, ncoords + ntets + 2):
        text = lines[i].split()
        v0, v1, v2, v3 = text[1], text[2], text[3], text[4]
        idolf_tets.append([int(v0) - 1, int(v1) - 1, int(v2) - 1, int(v3) - 1])

    if outfile:  # write dolfin xml
        outF = open(outfile, "w")
        outF.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        outF.write('<dolfin xmlns:dolfin="http://www.fenicsproject.org">\n')
        outF.write('  <mesh celltype="tetrahedron" dim="3">\n')

        outF.write('    <vertices size="' + str(ncoords) + '">\n')
        for i in range(ncoords):
            x, y, z = fdolf_coords[i]
            outF.write('      <vertex index="'+str(i)
                       + '" x="'+str(x)+'" y="'+str(y)+'" z="'+str(z)+'"/>\n')
        outF.write('    </vertices>\n')

        outF.write('    <cells size="' + str(ntets) + '">\n')
        for i in range(ntets):
            v0, v1, v2, v3 = idolf_tets[i]
            outF.write('      <tetrahedron index="'+str(i)
                       + '" v0="'+str(v0)+'" v1="'+str(v1)+'" v2="'+str(v2)+'" v3="'+str(v3)+'"/>\n')
        outF.write('    </cells>\n')

        outF.write("  </mesh>\n")
        outF.write("</dolfin>\n")
        outF.close()
    return fdolf_coords, idolf_tets


def buildPolyData(vertices, faces=None, indexOffset=0):
    """
    Build a ``vtkPolyData`` object from a list of vertices
    where faces represents the connectivity of the polygonal mesh.

    E.g. :
        - ``vertices=[[x1,y1,z1],[x2,y2,z2], ...]``
        - ``faces=[[0,1,2], [1,2,3], ...]``

    Use ``indexOffset=1`` if face numbering starts from 1 instead of 0.

    .. hint:: |buildpolydata| |buildpolydata.py|_
    """

    if not utils.isSequence(vertices):  # assume a dolfin.Mesh
        faces = vertices.cells()
        vertices = vertices.coordinates()

    sourcePoints = vtk.vtkPoints()
    sourcePolygons = vtk.vtkCellArray()
    sourceVertices = vtk.vtkCellArray()
    isgt2 = len(vertices[0]) > 2
    is1 = len(vertices[0]) == 1
    for pt in vertices:
        if isgt2:
            aid = sourcePoints.InsertNextPoint(pt[0], pt[1], pt[2])
        elif is1:
            aid = sourcePoints.InsertNextPoint(pt[0], 0, 0)
        else:
            aid = sourcePoints.InsertNextPoint(pt[0], pt[1], 0)
            
        if faces is None:
            sourceVertices.InsertNextCell(1)
            sourceVertices.InsertCellPoint(aid)
        
    if faces is not None:
        showbar = False
        if len(faces) > 25000:
            showbar = True
            pb = ProgressBar(0, len(faces), ETA=False)
        for f in faces:
            n = len(f)
            if n == 4:
                ele0 = vtk.vtkTriangle()
                ele1 = vtk.vtkTriangle()
                ele2 = vtk.vtkTriangle()
                ele3 = vtk.vtkTriangle()
                f0, f1, f2, f3 = f
                if indexOffset:  # for speed..
                    ele0.GetPointIds().SetId(0, f0 - indexOffset)
                    ele0.GetPointIds().SetId(1, f1 - indexOffset)
                    ele0.GetPointIds().SetId(2, f2 - indexOffset)

                    ele1.GetPointIds().SetId(0, f0 - indexOffset)
                    ele1.GetPointIds().SetId(1, f1 - indexOffset)
                    ele1.GetPointIds().SetId(2, f3 - indexOffset)

                    ele2.GetPointIds().SetId(0, f1 - indexOffset)
                    ele2.GetPointIds().SetId(1, f2 - indexOffset)
                    ele2.GetPointIds().SetId(2, f3 - indexOffset)

                    ele3.GetPointIds().SetId(0, f2 - indexOffset)
                    ele3.GetPointIds().SetId(1, f3 - indexOffset)
                    ele3.GetPointIds().SetId(2, f0 - indexOffset)
                else:
                    ele0.GetPointIds().SetId(0, f0)
                    ele0.GetPointIds().SetId(1, f1)
                    ele0.GetPointIds().SetId(2, f2)

                    ele1.GetPointIds().SetId(0, f0)
                    ele1.GetPointIds().SetId(1, f1)
                    ele1.GetPointIds().SetId(2, f3)

                    ele2.GetPointIds().SetId(0, f1)
                    ele2.GetPointIds().SetId(1, f2)
                    ele2.GetPointIds().SetId(2, f3)

                    ele3.GetPointIds().SetId(0, f2)
                    ele3.GetPointIds().SetId(1, f3)
                    ele3.GetPointIds().SetId(2, f0)

                sourcePolygons.InsertNextCell(ele0)
                sourcePolygons.InsertNextCell(ele1)
                sourcePolygons.InsertNextCell(ele2)
                sourcePolygons.InsertNextCell(ele3)

            elif n == 3:
                ele = vtk.vtkTriangle()
                for i in range(3):
                    ele.GetPointIds().SetId(i, f[i] - indexOffset)
                sourcePolygons.InsertNextCell(ele)

            else:
                ele = vtk.vtkPolygon()
                ele.GetPointIds().SetNumberOfIds(n)
                for i in range(n):
                    ele.GetPointIds().SetId(i, f[i] - indexOffset)
                sourcePolygons.InsertNextCell(ele)
            if showbar:
                pb.print("converting mesh..")

    poly = vtk.vtkPolyData()
    poly.SetPoints(sourcePoints)
    if faces is None:
        poly.SetVerts(sourceVertices)
    else:
        poly.SetPolys(sourcePolygons)
        
    return poly


##########################################################
def screenshot(filename="screenshot.png"):
    """
    Save a screenshot of the current rendering window.
    """
    if not settings.plotter_instance.window:
        colors.printc('~bomb screenshot(): Rendering window is not present, skip.', c=1)
        return
    w2if = vtk.vtkWindowToImageFilter()
    w2if.ShouldRerenderOff()
    w2if.SetInput(settings.plotter_instance.window)
    w2if.ReadFrontBufferOff()  # read from the back buffer
    w2if.Update()
    pngwriter = vtk.vtkPNGWriter()
    pngwriter.SetFileName(filename)
    pngwriter.SetInputConnection(w2if.GetOutputPort())
    pngwriter.Write()


class Video:
    """
    Class to generate a video from the specified rendering window.
    Only tested on linux systems with ``ffmpeg`` installed.

    :param str name: name of the output file.
    :param int fps: set the number of frames per second.
    :param float duration: set the total `duration` of the video and recalculates `fps` accordingly.

    .. hint:: |makeVideo| |makeVideo.py|_
    """

    def __init__(self, name="movie.avi", fps=12, duration=None):

        import glob

        self.name = name
        self.duration = duration
        self.fps = float(fps)
        self.frames = []
        if not os.path.exists("/tmp/vpvid"):
            os.mkdir("/tmp/vpvid")
        for fl in glob.glob("/tmp/vpvid/*.png"):
            os.remove(fl)
        colors.printc("~video Video", name, "is open...", c="m")

    def addFrame(self):
        """Add frame to current video."""
        fr = "/tmp/vpvid/" + str(len(self.frames)) + ".png"
        screenshot(fr)
        self.frames.append(fr)

    def pause(self, pause=0):
        """Insert a `pause`, in seconds."""
        fr = self.frames[-1]
        n = int(self.fps * pause)
        for i in range(n):
            fr2 = "/tmp/vpvid/" + str(len(self.frames)) + ".png"
            self.frames.append(fr2)
            os.system("cp -f %s %s" % (fr, fr2))

    def close(self):
        """Render the video and write to file."""
        if self.duration:
            _fps = len(self.frames) / float(self.duration)
            colors.printc("Recalculated video FPS to", round(_fps, 3), c="yellow")
        else:
            _fps = int(_fps)
        self.name = self.name.split('.')[0]+'.mp4'
        out = os.system("ffmpeg -loglevel panic -y -r " + str(_fps)
                        + " -i /tmp/vpvid/%01d.png "+self.name)
        if out:
            colors.printc("ffmpeg returning error", c=1)
        colors.printc("~save Video saved as", self.name, c="green")
        return


###########################################################################
class ProgressBar:
    """
    Class to print a progress bar with optional text message.

    :Example:
        .. code-block:: python
        
            import time
            pb = ProgressBar(0,400, c='red')
            for i in pb.range():
                time.sleep(.1)
                pb.print('some message') # or pb.print(counts=i)

        |progbar|
    """

    def __init__(self, start, stop, step=1, c=None, ETA=True, width=24, char="="):

        self.start = start
        self.stop = stop
        self.step = step
        self.color = c
        self.width = width
        self.char = char
        self.bar = ""
        self.percent = 0
        self.clock0 = 0
        self.ETA = ETA
        self.clock0 = time.time()
        self._remt = 1e10
        self._update(0)
        self._counts = 0
        self._oldbar = ""
        self._lentxt = 0
        self._range = numpy.arange(start, stop, step)
        self._len = len(self._range)

    def print(self, txt="", counts=None):
        if counts:
            self._update(counts)
        else:
            self._update(self._counts + self.step)
        if self.bar != self._oldbar:
            self._oldbar = self.bar
            eraser = [" "] * self._lentxt + ["\b"] * self._lentxt
            eraser = "".join(eraser)
            if self.ETA:
                vel = self._counts / (time.time() - self.clock0)
                self._remt = (self.stop - self._counts) / vel
                if self._remt > 60:
                    mins = int(self._remt / 60)
                    secs = self._remt - 60 * mins
                    mins = str(mins) + "m"
                    secs = str(int(secs + 0.5)) + "s "
                else:
                    mins = ""
                    secs = str(int(self._remt + 0.5)) + "s "
                vel = str(round(vel, 1))
                eta = "ETA: " + mins + secs + "(" + vel + " it/s) "
                if self._remt < 1:
                    dt = time.time() - self.clock0
                    if dt > 60:
                        mins = int(dt / 60)
                        secs = dt - 60 * mins
                        mins = str(mins) + "m"
                        secs = str(int(secs + 0.5)) + "s "
                    else:
                        mins = ""
                        secs = str(int(dt + 0.5)) + "s "
                    eta = "Elapsed time: " + mins + secs + "(" + vel + " it/s)        "
                    txt = ""
            else:
                eta = ""
            txt = eta + str(txt)
            s = self.bar + " " + eraser + txt + "\r"
            if self.color:
                colors.printc(s, c=self.color, end="")
            else:
                sys.stdout.write(s)
                sys.stdout.flush()
            if self.percent == 100:
                print("")
            self._lentxt = len(txt)

    def range(self):
        return self._range

    def len(self):
        return self._len

    def _update(self, counts):
        if counts < self.start:
            counts = self.start
        elif counts > self.stop:
            counts = self.stop
        self._counts = counts
        self.percent = (self._counts - self.start) * 100
        self.percent /= self.stop - self.start
        self.percent = int(round(self.percent))
        af = self.width - 2
        nh = int(round(self.percent / 100 * af))
        if nh == 0:
            self.bar = "[>%s]" % (" " * (af - 1))
        elif nh == af:
            self.bar = "[%s]" % (self.char * af)
        else:
            self.bar = "[%s>%s]" % (self.char * (nh - 1), " " * (af - nh))
        if self.percent < 100:  # and self._remt > 1:
            ps = " " + str(self.percent) + "%"
        else:
            ps = ""
        self.bar += ps


#############
class Button:
    """
    Build a Button object to be shown in the rendering window.

    .. hint:: |buttons| |buttons.py|_
    """

    def __init__(self, fnc, states, c, bc, pos, size, font, bold, italic, alpha, angle):
        """
        Build a Button object to be shown in the rendering window.
        """
        self._status = 0
        self.states = states
        self.colors = c
        self.bcolors = bc
        self.function = fnc
        self.actor = vtk.vtkTextActor()
        self.actor.SetDisplayPosition(pos[0], pos[1])
        self.framewidth = 3
        self.offset = 5
        self.spacer = " "

        self.textproperty = self.actor.GetTextProperty()
        self.textproperty.SetJustificationToCentered()
        if font.lower() == "courier":
            self.textproperty.SetFontFamilyToCourier()
        elif font.lower() == "times":
            self.textproperty.SetFontFamilyToTimes()
        else:
            self.textproperty.SetFontFamilyToArial()
        self.textproperty.SetFontSize(size)
        self.textproperty.SetBackgroundOpacity(alpha)
        self.textproperty.BoldOff()
        if bold:
            self.textproperty.BoldOn()
        self.textproperty.ItalicOff()
        if italic:
            self.textproperty.ItalicOn()
        self.textproperty.ShadowOff()
        self.textproperty.SetOrientation(angle)
        self.showframe = hasattr(self.textproperty, "FrameOn")
        self.status(0)

    def status(self, s=None):
        """
        Set/Get the status of the button.
        """
        if s is None:
            return self.states[self._status]
        if isinstance(s, str):
            s = self.states.index(s)
        self._status = s
        self.textproperty.SetLineOffset(self.offset)
        self.actor.SetInput(self.spacer + self.states[s] + self.spacer)
        s = s % len(self.colors)  # to avoid mismatch
        self.textproperty.SetColor(colors.getColor(self.colors[s]))
        bcc = numpy.array(colors.getColor(self.bcolors[s]))
        self.textproperty.SetBackgroundColor(bcc)
        if self.showframe:
            self.textproperty.FrameOn()
            self.textproperty.SetFrameWidth(self.framewidth)
            self.textproperty.SetFrameColor(numpy.sqrt(bcc))

    def switch(self):
        """
        Change/cycle button status to the next defined status in states list.
        """
        self._status = (self._status + 1) % len(self.states)
        self.status(self._status)


# ############################################################### Mouse Events
def _mouse_enter(iren, event):

    #x, y = iren.GetEventPosition()
    #print('_mouse_enter mouse at', x, y)

    for ivp in settings.plotter_instances:
        if ivp.interactor != iren:
            if ivp.camera == iren.GetActiveCamera():
                ivp.interactor.Render()
        

def _mouseleft(iren, event):

    x, y = iren.GetEventPosition()
    #print('_mouseleft mouse at', x, y)

    renderer = iren.FindPokedRenderer(x, y)

    vp = None
    for ivp in settings.plotter_instances:
        if renderer in ivp.renderers:
            vp = ivp
            break
    if not vp:
        return

    vp.renderer = renderer
    
    picker = vtk.vtkPropPicker()
    picker.PickProp(x, y, renderer)
    clickedActor = picker.GetActor()

    # check if any button objects are clicked
    clickedActor2D = picker.GetActor2D()
    if clickedActor2D:
        for bt in vp.buttons:
            if clickedActor2D == bt.actor:
                bt.function()
                break

    if not clickedActor:
        clickedActor = picker.GetAssembly()
    vp.picked3d = picker.GetPickPosition()
    vp.justremoved = None

    if not hasattr(clickedActor, "GetPickable") or not clickedActor.GetPickable():
        return
    vp.clickedActor = clickedActor

    if vp.mouseLeftClickFunction:
        vp.mouseLeftClickFunction(clickedActor)
    

def _mouseright(iren, event):

    x, y = iren.GetEventPosition()

    renderer = iren.FindPokedRenderer(x, y)
    vp = None
    for ivp in settings.plotter_instances:
        if renderer in ivp.renderers:
            vp = ivp
            break
    if not vp:
        return

    vp.renderer = renderer

    picker = vtk.vtkPropPicker()
    picker.PickProp(x, y, renderer)
    clickedActor = picker.GetActor()

    # check if any button objects were created
    clickedActor2D = picker.GetActor2D()
    if clickedActor2D:
        for bt in vp.buttons:
            if clickedActor2D == bt.actor:
                bt.function()
                break

    if not clickedActor:
        clickedActor = picker.GetAssembly()
    vp.picked3d = picker.GetPickPosition()

    if not hasattr(clickedActor, "GetPickable") or not clickedActor.GetPickable():
        return
    vp.clickedActor = clickedActor

    if vp.mouseRightClickFunction:
        vp.mouseRightClickFunction(clickedActor)


def _mousemiddle(iren, event):

    x, y = iren.GetEventPosition()

    renderer = iren.FindPokedRenderer(x, y)
    vp = None
    for ivp in settings.plotter_instances:
        if renderer in ivp.renderers:
            vp = ivp
            break
    if not vp:
        return

    vp.renderer = renderer
    
    picker = vtk.vtkPropPicker()
    picker.PickProp(x, y, renderer)
    clickedActor = picker.GetActor()

    # check if any button objects were created
    clickedActor2D = picker.GetActor2D()
    if clickedActor2D:
        for bt in vp.buttons:
            if clickedActor2D == bt.actor:
                bt.function()
                break

    if not clickedActor:
        clickedActor = picker.GetAssembly()
    vp.picked3d = picker.GetPickPosition()

    if not hasattr(clickedActor, "GetPickable") or not clickedActor.GetPickable():
        return
    vp.clickedActor = clickedActor

    if vp.mouseMiddleClickFunction:
        vp.mouseMiddleClickFunction(vp.clickedActor)


def _keypress(iren, event):
    # qt creates and passes a vtkGenericRenderWindowInteractor

    vp = settings.plotter_instance
    key = iren.GetKeySym()
    # print('Pressed key:', key, event)

    if key in ["q", "Q", "space", "Return"]:
        iren.ExitCallback()
        return

    elif key == "e":
        if vp.verbose:
            print("closing window...")
        iren.GetRenderWindow().Finalize()
        iren.TerminateApp()
        return

    elif key == "Escape":
        print()
        sys.stdout.flush()
        iren.TerminateApp()
        iren.GetRenderWindow().Finalize()
        iren.TerminateApp()
        sys.exit(0)

    elif key == "m":
        if vp.clickedActor in vp.getActors():
            vp.clickedActor.GetProperty().SetOpacity(0.02)
            bfp = vp.clickedActor.GetBackfaceProperty()
            if bfp and hasattr(vp.clickedActor, "_bfprop"):
                vp.clickedActor._bfprop = bfp  # save it
                vp.clickedActor.SetBackfaceProperty(None)
        else:
            for a in vp.getActors():
                if a.GetPickable():
                    a.GetProperty().SetOpacity(0.02)
                    bfp = a.GetBackfaceProperty()
                    if bfp and hasattr(a, "_bfprop"):
                        a._bfprop = bfp
                        a.SetBackfaceProperty(None)

    elif key == "comma":
        if vp.clickedActor in vp.getActors():
            ap = vp.clickedActor.GetProperty()
            aal = max([ap.GetOpacity() * 0.75, 0.01])
            ap.SetOpacity(aal)
            bfp = vp.clickedActor.GetBackfaceProperty()
            if bfp and hasattr(vp.clickedActor, "_bfprop"):
                vp.clickedActor._bfprop = bfp
                vp.clickedActor.SetBackfaceProperty(None)
        else:
            for a in vp.getActors():
                if a.GetPickable():
                    ap = a.GetProperty()
                    aal = max([ap.GetOpacity() * 0.75, 0.01])
                    ap.SetOpacity(aal)
                    bfp = a.GetBackfaceProperty()
                    if bfp and hasattr(a, "_bfprop"):
                        a._bfprop = bfp
                        a.SetBackfaceProperty(None)

    elif key == "period":
        if vp.clickedActor in vp.getActors():
            ap = vp.clickedActor.GetProperty()
            aal = min([ap.GetOpacity() * 1.25, 1.0])
            ap.SetOpacity(aal)
            if aal == 1 and hasattr(vp.clickedActor, "_bfprop") and vp.clickedActor._bfprop:
                # put back
                vp.clickedActor.SetBackfaceProperty(vp.clickedActor._bfprop)
        else:
            for a in vp.getActors():
                if a.GetPickable():
                    ap = a.GetProperty()
                    aal = min([ap.GetOpacity() * 1.25, 1.0])
                    ap.SetOpacity(aal)
                    if aal == 1 and hasattr(a, "_bfprop") and a._bfprop:
                        a.SetBackfaceProperty(a._bfprop)

    elif key == "slash":
        if vp.clickedActor in vp.getActors():
            vp.clickedActor.GetProperty().SetOpacity(1)
            if hasattr(vp.clickedActor, "_bfprop") and vp.clickedActor._bfprop:
                vp.clickedActor.SetBackfaceProperty(vp.clickedActor._bfprop)
        else:
            for a in vp.getActors():
                if a.GetPickable():
                    a.GetProperty().SetOpacity(1)
                    if hasattr(a, "_bfprop") and a._bfprop:
                        a.clickedActor.SetBackfaceProperty(a._bfprop)

    elif key == "P":
        if vp.clickedActor in vp.getActors():
            acts = [vp.clickedActor]
        else:
            acts = vp.getActors()
        for ia in acts:
            if ia.GetPickable():
                try:
                    ps = ia.GetProperty().GetPointSize()
                    if ps > 1:
                        ia.GetProperty().SetPointSize(ps - 1)
                    ia.GetProperty().SetRepresentationToPoints()
                except AttributeError:
                    pass

    elif key == "p":
        if vp.clickedActor in vp.getActors():
            acts = [vp.clickedActor]
        else:
            acts = vp.getActors()
        for ia in acts:
            if ia.GetPickable():
                try:
                    ps = ia.GetProperty().GetPointSize()
                    ia.GetProperty().SetPointSize(ps + 2)
                    ia.GetProperty().SetRepresentationToPoints()
                except AttributeError:
                    pass

    elif key == "w":
        if vp.clickedActor and vp.clickedActor in vp.getActors():
            vp.clickedActor.GetProperty().SetRepresentationToWireframe()
        else:
            for a in vp.getActors():
                if a and a.GetPickable():
                    if a.GetProperty().GetRepresentation() == 1:  # toggle
                        a.GetProperty().SetRepresentationToSurface()
                    else:
                        a.GetProperty().SetRepresentationToWireframe()

    elif key == "r":
        vp.renderer.ResetCamera()

    #############################################################
    ### now intercept custom observer ###########################
    #############################################################
    if vp.keyPressFunction:
        if key not in ["Shift_L", "Control_L", "Super_L", "Alt_L"]:
            if key not in ["Shift_R", "Control_R", "Super_R", "Alt_R"]:
                vp.verbose = False
                vp.keyPressFunction(key)
                return

    if key == "h":
        from vtkplotter.docs import tips

        tips()
        return

    if key == "a":
        iren.ExitCallback()
        cur = iren.GetInteractorStyle()
        if isinstance(cur, vtk.vtkInteractorStyleTrackballCamera):
            print("\nInteractor style changed to TrackballActor")
            print("  you can now move and rotate individual meshes:")
            print("  press X twice to save the repositioned mesh,")
            print("  press 'a' to go back to normal style.")
            iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballActor())
        else:
            iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        iren.Start()
        return

    if key == "j":
        iren.ExitCallback()
        cur = iren.GetInteractorStyle()
        if isinstance(cur, vtk.vtkInteractorStyleJoystickCamera):
            iren.SetInteractorStyle(vtk.vtkInteractorStyleTrackballCamera())
        else:
            print("\nInteractor style changed to Joystick,", end="")
            print(" press j to go back to normal.")
            iren.SetInteractorStyle(vtk.vtkInteractorStyleJoystickCamera())
        iren.Start()
        return

    if key == "S":
        screenshot("screenshot.png")
        colors.printc("~camera Saved rendering window as screenshot.png", c="blue")
        return

    elif key == "C":
        cam = vp.renderer.GetActiveCamera()
        print('\n### Example code to position this vtkCamera:')
        print('vp = vtkplotter.Plotter()\n...')
        print('vp.camera.SetPosition(',   [round(e, 3) for e in cam.GetPosition()],  ')')
        print('vp.camera.SetFocalPoint(', [round(e, 3) for e in cam.GetFocalPoint()], ')')
        print('vp.camera.SetViewUp(',     [round(e, 3) for e in cam.GetViewUp()], ')')
        print('vp.camera.SetDistance(',   round(cam.GetDistance(), 3), ')')
        print('vp.camera.SetClippingRange(', [round(e, 3) for e in cam.GetClippingRange()], ')')
        return

    elif key == "s":
        if vp.clickedActor and vp.clickedActor in vp.getActors():
            vp.clickedActor.GetProperty().SetRepresentationToSurface()
        else:
            for a in vp.getActors():
                if a and a.GetPickable():
                    a.GetProperty().SetRepresentationToSurface()

    elif key == "V":
        if not (vp.verbose):
            vp._tips()
        vp.verbose = not (vp.verbose)
        print("Verbose: ", vp.verbose)

    elif key == "1":
        if vp.clickedActor and hasattr(vp.clickedActor, "GetProperty"):
            vp.clickedActor.GetProperty().SetColor(colors.colors1[(vp.icol) % 10])
        else:
            for i, ia in enumerate(vp.getActors()):
                if not ia.GetPickable():
                    continue
                ia.GetProperty().SetColor(colors.colors1[(i + vp.icol) % 10])
                ia.GetMapper().ScalarVisibilityOff()
        vp.icol += 1
        addons.addLegend()

    elif key == "2":
        if vp.clickedActor and hasattr(vp.clickedActor, "GetProperty"):
            vp.clickedActor.GetProperty().SetColor(colors.colors2[(vp.icol) % 10])
        else:
            for i, ia in enumerate(vp.getActors()):
                if not ia.GetPickable():
                    continue
                ia.GetProperty().SetColor(colors.colors2[(i + vp.icol) % 10])
                ia.GetMapper().ScalarVisibilityOff()
        vp.icol += 1
        addons.addLegend()

    elif key == "3":
        c = colors.getColor("gold")
        acs = vp.getActors()
        alpha = 1.0 / len(acs)
        for ia in acs:
            if not ia.GetPickable():
                continue
            ia.GetProperty().SetColor(c)
            ia.GetProperty().SetOpacity(alpha)
            ia.GetMapper().ScalarVisibilityOff()
        addons.addLegend()

    elif key == "4":
        bgc = numpy.array(vp.renderer.GetBackground()).sum() / 3
        if bgc > 1:
            bgc = -0.223
        vp.renderer.SetBackground(bgc + 0.223, bgc + 0.223, bgc + 0.223)

    elif "KP_" in key:  # change axes style
        asso = {
                "KP_Insert":0, "KP_0":0,
                "KP_End":1,    "KP_1":1,
                "KP_Down":2,   "KP_2":2,
                "KP_Next":3,   "KP_3":3,
                "KP_Left":4,   "KP_4":4,
                "KP_Begin":5,  "KP_5":5,
                "KP_Right":6,  "KP_6":6,
                "KP_Home":7,   "KP_7":7,
                "KP_Up":8,     "KP_8":8,
                "KP_Prior":9,  "KP_9":9,
                }
        clickedr = vp.renderers.index(vp.renderer)
        if key in asso.keys():
            if vp.axes_exist[clickedr]:
                if hasattr(vp.axes_exist[clickedr], "EnabledOff"):  # widget
                    vp.axes_exist[clickedr].EnabledOff()
                else:
                    vp.renderer.RemoveActor(vp.axes_exist[clickedr])
                vp.axes_exist[clickedr] = None
            addons.addAxes(axtype=asso[key], c=None)
            vp.interactor.Render()

    elif key in ["k", "K"]:
        for a in vp.getActors():
            ptdata = a.polydata().GetPointData()
            cldata = a.polydata().GetCellData()

            arrtypes = dict()
            arrtypes[vtk.VTK_UNSIGNED_CHAR] = "VTK_UNSIGNED_CHAR"
            arrtypes[vtk.VTK_UNSIGNED_INT] = "VTK_UNSIGNED_INT"
            arrtypes[vtk.VTK_FLOAT] = "VTK_FLOAT"
            arrtypes[vtk.VTK_DOUBLE] = "VTK_DOUBLE"
            foundarr = 0

            if key == "k":
                for i in range(ptdata.GetNumberOfArrays()):
                    name = ptdata.GetArrayName(i)
                    if name == "Normals":
                        continue
                    ptdata.SetActiveScalars(name)
                    foundarr = 1
                if not foundarr:
                    print("No vtkArray is associated to points", end="")
                    if hasattr(a, "_legend"):
                        print(" for actor:", a._legend)
                    else:
                        print()

            if key == "K":
                for i in range(cldata.GetNumberOfArrays()):
                    name = cldata.GetArrayName(i)
                    if name == "Normals":
                        continue
                    cldata.SetActiveScalars(name)
                    foundarr = 1
                if not foundarr:
                    print("No vtkArray is associated to cells", end="")
                    if hasattr(a, "_legend"):
                        print(" for actor:", a._legend)
                    else:
                        print()

            a.GetMapper().ScalarVisibilityOn()

    elif key == "L":
        if vp.clickedActor in vp.getActors():
            acts = [vp.clickedActor]
        else:
            acts = vp.getActors()
        for ia in acts:
            if not ia.GetPickable():
                continue
            try:
                ia.GetProperty().SetRepresentationToSurface()
                ls = ia.GetProperty().GetLineWidth()
                if ls <= 1:
                    ls = 1
                    ia.GetProperty().EdgeVisibilityOff()
                else:
                    ia.GetProperty().EdgeVisibilityOn()
                    ia.GetProperty().SetLineWidth(ls - 1)
            except AttributeError:
                pass

    elif key == "l":
        if vp.clickedActor in vp.getActors():
            acts = [vp.clickedActor]
        else:
            acts = vp.getActors()
        for ia in acts:
            if not ia.GetPickable():
                continue
            try:
                ia.GetProperty().EdgeVisibilityOn()
                c = ia.GetProperty().GetColor()
                ia.GetProperty().SetEdgeColor(c)
                ls = ia.GetProperty().GetLineWidth()
                ia.GetProperty().SetLineWidth(ls + 1)
            except AttributeError:
                pass

    elif key == "n":  # show normals to an actor
        from vtkplotter.analysis import normalLines

        if vp.clickedActor in vp.getActors():
            if vp.clickedActor.GetPickable():
                vp.renderer.AddActor(normalLines(vp.clickedActor))
                iren.Render()
        else:
            print("Click an actor and press n to add normals.")


    elif key == "x":
        if vp.justremoved is None:
            if vp.clickedActor in vp.getActors() or isinstance(vp.clickedActor, vtk.vtkAssembly):
                vp.justremoved = vp.clickedActor
                vp.renderer.RemoveActor(vp.clickedActor)
            if hasattr(vp.clickedActor, '_legend') and vp.clickedActor._legend:
                print('...removing actor: ' +
                      str(vp.clickedActor._legend)+', press x to put it back')
            else:
                print("Click an actor and press x to toggle it.")
        else:
            vp.renderer.AddActor(vp.justremoved)
            vp.renderer.Render()
            vp.justremoved = None
        addons.addLegend()

    elif key == "X":
        if vp.clickedActor:
            if not vp.cutterWidget:
                addons.addCutterTool(vp.clickedActor)
            else:
                fname = "clipped.vtk"
                confilter = vtk.vtkPolyDataConnectivityFilter()
                if isinstance(vp.clickedActor, vtk.vtkActor):
                    confilter.SetInputData(vp.clickedActor.GetMapper().GetInput())
                elif isinstance(vp.clickedActor, vtk.vtkAssembly):
                    act = vp.clickedActor.getActors()[0]
                    confilter.SetInputData(act.GetMapper().GetInput())
                else:
                    confilter.SetInputData(vp.clickedActor.polydata(True))
                confilter.SetExtractionModeToLargestRegion()
                confilter.Update()
                cpd = vtk.vtkCleanPolyData()
                cpd.SetInputData(confilter.GetOutput())
                cpd.Update()
                w = vtk.vtkPolyDataWriter()
                w.SetInputData(cpd.GetOutput())
                w.SetFileName(fname)
                w.Write()
                colors.printc("~save Saved file:", fname, c="m")
                vp.cutterWidget.Off()
                vp.cutterWidget = None
        else:
            for a in vp.actors:
                if isinstance(a, vtk.vtkVolume):
                    addons.addCutterTool(a)
                    return

            colors.printc("Click an actor and press X to open the cutter box widget.", c=4)

    elif key == "i":  # print info
        if vp.clickedActor:
            utils.printInfo(vp.clickedActor)
        else:
            utils.printInfo(vp)

    if iren:
        iren.Render()
