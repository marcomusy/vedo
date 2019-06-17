from __future__ import division, print_function
import vtk
import os
import sys
import numpy

import vtkplotter.utils as utils
import vtkplotter.colors as colors
from vtkplotter.actors import Actor, Volume, Assembly, Image, isosurface
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
    "download",
    "gunzip",
    "loadStructuredPoints",
    "loadStructuredGrid",
    "loadUnStructuredGrid",
    "loadRectilinearGrid",
    "write",
    "save",
    "exportWindow",
    "screenshot",
    "Video",
]


def load(inputobj, c="gold", alpha=1, threshold=False, spacing=(), unpack=True):
    """
    Load ``Actor`` and ``Volume`` from file.

    The output will depend on the file extension. See examples below.

    :param c: color in RGB format, hex, symbol or name
    :param alpha: transparency/opacity of the polygonal data.

    For volumetric data `(tiff, slc, vti etc..)`:

    :param list c: can be a list of any length of colors. This list represents the color
        transfer function values equally spaced along the range of the volumetric scalar.
    :param list alpha: can be a list of any length of tranparencies. This list represents the
        transparency transfer function values equally spaced along the range of the volumetric scalar.
    :param float threshold: value to draw the isosurface, False by default to return a ``Volume``.
        If set to True will return an ``Actor`` with automatic choice of the isosurfacing threshold.
    :param list spacing: specify the voxel spacing in the three dimensions
    :param bool unpack: only for multiblock data, if True returns a flat list of objects.

    :Examples:
        .. code-block:: python

            from vtkplotter import datadir, load, show

            # Return an Actor
            g = load(datadir+'250.vtk')
            show(g)

            # Return a list of 2 Actors
            g = load([datadir+'250.vtk', datadir+'270.vtk'])
            show(g)

            # Return a list of actors by reading all files in a directory
            # (if directory contains DICOM files then a Volume is returned)
            g = load(datadir+'timecourse1d/')
            show(g)

            # Return a Volume. Color/Opacity transfer functions can be specified too.
            g = load(datadir+'embryo.slc')
            g.c(['y','lb','w']).alpha((0.0, 0.4, 0.9, 1))
            show(g)

            # Return an Actor from a SLC volume with automatic thresholding
            g = load(datadir+'embryo.slc', threshold=True)
            show(g)
    """
    acts = []
    if utils.isSequence(inputobj):
        flist = inputobj
    else:
        import glob
        flist = sorted(glob.glob(inputobj))

    for fod in flist:

        if os.path.isfile(fod): ### it's a file

            if fod.endswith("wrl"):
                importer = vtk.vtkVRMLImporter()
                importer.SetFileName(fod)
                importer.Read()
                importer.Update()
                actors = importer.GetRenderer().GetActors() #vtkActorCollection
                actors.InitTraversal()
                for i in range(actors.GetNumberOfItems()):
                    act = actors.GetNextActor()
                    acts.append(act)
            else:
                a = _load_file(fod, c, alpha, threshold, spacing, unpack)
                acts.append(a)

        elif os.path.isdir(fod):### it's a directory or DICOM
            flist = os.listdir(fod)
            if '.dcm' in flist[0]: ### it's DICOM
                reader = vtk.vtkDICOMImageReader()
                reader.SetDirectoryName(fod)
                reader.Update()
                image = reader.GetOutput()
                if len(spacing) == 3:
                    image.SetSpacing(spacing[0], spacing[1], spacing[2])
                if threshold is False:
                    if c is "gold" and alpha is 1:
                        c = ['b','lb','lg','y','r'] # good for blackboard background
                        alpha = (0.0, 0.0, 0.2, 0.4, 0.8, 1)
                        #c = ['lb','db','dg','dr']  # good for white backgr
                        #alpha = (0.0, 0.0, 0.2, 0.6, 0.8, 1)
                    actor = Volume(image, c, alpha)
                else:
                    actor = isosurface(image, threshold=threshold)
                    actor.color(c).alpha(alpha)
                acts.append(actor)
            else: ### it's a normal directory
                utils.humansort(flist)
                for ifile in flist:
                    a = _load_file(fod+'/'+ifile, c, alpha, threshold, spacing, unpack)
                    acts.append(a)
        else:
            colors.printc("~times Error in load(): cannot find", fod, c=1)

    if len(acts) == 1:
        if not acts[0]:
            colors.printc("~times Error in load(): cannot find", inputobj, c=1)
        return acts[0]
    elif len(acts) == 0:
        colors.printc("~times Error in load(): cannot find", inputobj, c=1)
        return None
    else:
        return acts


def _load_file(filename, c, alpha, threshold, spacing, unpack):
    fl = filename.lower()

    ################################################################# other formats:
    if fl.endswith(".xml") or fl.endswith(".xml.gz"):  # Fenics tetrahedral file
        actor = loadDolfin(filename)
    elif fl.endswith(".neutral") or fl.endswith(".neu"):  # neutral tetrahedral file
        actor = loadNeutral(filename)
    elif fl.endswith(".gmsh"):  # gmesh file
        actor = loadGmesh(filename)
    elif fl.endswith(".pcd"):  # PCL point-cloud format
        actor = loadPCD(filename)
        actor.GetProperty().SetPointSize(2)
    elif fl.endswith(".off"):
        actor = loadOFF(filename)
    elif fl.endswith(".3ds"):  # 3ds format
        actor = load3DS(filename)

        ################################################################# volumetric:
    elif fl.endswith(".tif") or fl.endswith(".slc") or fl.endswith(".vti") \
        or fl.endswith(".mhd") or fl.endswith(".nrrd") or fl.endswith(".nii"):
        img = loadImageData(filename, spacing)
        if threshold is False:
            if c is "gold" and alpha is 1:
                c = ['b','lb','lg','y','r'] # good for blackboard background
                alpha = (0.0, 0.0, 0.2, 0.4, 0.8, 1)
                #c = ['lb','db','dg','dr']  # good for white backgr
                #alpha = (0.0, 0.0, 0.2, 0.6, 0.8, 1)
            actor = Volume(img, c, alpha)
        else:
            actor = isosurface(img, threshold=threshold)
            actor.color(c).alpha(alpha)

        ################################################################# 2D images:
    elif fl.endswith(".png") or fl.endswith(".jpg") or fl.endswith(".bmp") or fl.endswith(".jpeg"):
        if ".png" in fl:
            picr = vtk.vtkPNGReader()
        elif ".jpg" in fl or ".jpeg" in fl:
            picr = vtk.vtkJPEGReader()
        elif ".bmp" in fl:
            picr = vtk.vtkBMPReader()
        picr.SetFileName(filename)
        picr.Update()
        actor = Image()  # object derived from vtk.vtkImageActor()
        actor.SetInputData(picr.GetOutput())
        if alpha is None:
            alpha = 1
        actor.SetOpacity(alpha)

        ################################################################# multiblock:
    elif fl.endswith(".vtm"):
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

    elif fl.endswith(".geojson") or fl.endswith(".geojson.gz"):
        return loadGeoJSON(fl)
    
        ################################################################# polygonal mesh:
    else:
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
        elif fl.endswith(".foam"):  # OpenFoam
            reader = vtk.vtkOpenFOAMReader()
        elif fl.endswith(".vtp"):
            reader = vtk.vtkXMLPolyDataReader()
        elif fl.endswith(".vts"):
            reader = vtk.vtkXMLStructuredGridReader()
        elif fl.endswith(".vtu"):
            reader = vtk.vtkXMLUnstructuredGridReader()
        elif fl.endswith(".txt"):
            reader = vtk.vtkParticleReader()  # (format is x, y, z, scalar)
        elif fl.endswith(".xyz"):
            reader = vtk.vtkParticleReader()
        elif fl.endswith(".pvtk"):
            reader = vtk.vtkPDataSetReader()
        elif fl.endswith(".pvtr"):
            reader = vtk.vtkXMLPRectilinearGridReader()
        elif fl.endswith("pvtu"):
            reader = vtk.vtkXMLPUnstructuredGridReader()
        elif fl.endswith(".pvti"):
            reader = vtk.vtkXMLPImageDataReader()
        else:
            reader = vtk.vtkDataReader()
        reader.SetFileName(filename)
        reader.Update()
        poly = reader.GetOutput()

        if fl.endswith(".vts") or fl.endswith(".vtu"): # un/structured grid
            gf = vtk.vtkGeometryFilter()
            gf.SetInputData(poly)
            gf.Update()
            poly = gf.GetOutput()

        if not poly:
            colors.printc("~noentry Unable to load", filename, c=1)
            return None

        actor = Actor(poly, c, alpha)
        if fl.endswith(".txt") or fl.endswith(".xyz"):
            actor.GetProperty().SetPointSize(4)

    actor.filename = filename
    return actor

def download(url, prefix=''):
    """Retrieve a file from a url, save it locally and return its path."""

    if "https://" not in url and "http://" not in url:
        colors.printc('Invalid URL:\n', url, c=1)
        return

    basename = os.path.basename(url)
    if os.path.exists(basename):
        return basename

    colors.printc('..downloading:\n', basename)
    try:
        from urllib.request import urlopen
    except ImportError:
        import urllib2
        import contextlib
        urlopen = lambda url_: contextlib.closing(urllib2.urlopen(url_))

    basename += prefix
    with urlopen(url) as response, open(basename, 'wb') as output:
        output.write(response.read())
    return basename

def gunzip(filename):
    """Unzip a ``.gz`` file to a temporary file and returns its path."""
    if not filename.endswith('.gz'):
        #colors.printc("gunzip() error: file must end with .gz", c=1)
        return filename
    from tempfile import NamedTemporaryFile
    import gzip

    tmp_file = NamedTemporaryFile(delete=False)
    tmp_file.name = os.path.join(os.path.dirname(tmp_file.name),
                                 os.path.basename(filename).replace('.gz',''))
    inF = gzip.open(filename, "rb")
    outF = open(tmp_file.name, "wb")
    outF.write(inF.read())
    outF.close()
    inF.close()
    return tmp_file.name


###################################################################
def loadStructuredPoints(filename):
    """Load a ``vtkStructuredPoints`` object from file."""
    reader = vtk.vtkStructuredPointsReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def loadStructuredGrid(filename):
    """Load a ``vtkStructuredGrid`` object from file."""
    reader = vtk.vtkStructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def loadUnStructuredGrid(filename):
    """Load a ``vtkunStructuredGrid`` object from file."""
    reader = vtk.vtkUnstructuredGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def loadRectilinearGrid(filename):
    """Load a ``vtkRectilinearGrid`` object from file."""
    reader = vtk.vtkRectilinearGridReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


def loadXMLGenericData(filename):
    """Read any type of vtk data object encoded in XML format."""
    reader = vtk.vtkXMLGenericDataObjectReader()
    reader.SetFileName(filename)
    reader.Update()
    return reader.GetOutput()


###################################################################
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


def loadOFF(filename):
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
            ids += [int(xx) for xx in ts[1:]]
            faces.append(ids)

    return Actor(buildPolyData(vertices, faces))


def loadGeoJSON(filename):
    """Load GeoJSON files."""
    if filename.endswith('.gz'):
        filename = gunzip(filename)
    jr = vtk.vtkGeoJSONReader()
    jr.SetFileName(filename)
    jr.Update()
    return Actor(jr.GetOutput())
    

def loadDolfin(filename):
    """Reads a `Fenics/Dolfin` file format. Return an ``Actor(vtkActor)`` object."""
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
    return Actor(poly, alpha=0.5)


def loadNeutral(filename):
    """Reads a `Neutral` tetrahedral file format. Return an ``Actor(vtkActor)`` object."""
    if not os.path.exists(filename):
        colors.printc("~noentry Error in loadNeutral: Cannot find", filename, c=1)
        return None
    f = open(filename, "r")
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

    poly = buildPolyData(fdolf_coords, idolf_tets, indexOffset=0)
    return Actor(poly)


def loadGmesh(filename):
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

    return Actor(poly)


def loadPCD(filename):
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
    actor = Actor(poly)
    actor.GetProperty().SetPointSize(4)
    return actor


def loadImageData(filename, spacing=()):
    """Read and return a ``vtkImageData`` object from file.
    Use ``loadVolume`` instead.
    E.g. `img = loadVolume('myfile.tif').imagedata()`
    """
    if not os.path.isfile(filename):
        colors.printc("~noentry File not found:", filename, c=1)
        return None

    if ".tif" in filename.lower():
        reader = vtk.vtkTIFFReader()
    elif ".slc" in filename.lower():
        reader = vtk.vtkSLCReader()
        if not reader.CanReadFile(filename):
            colors.printc("~prohibited Sorry bad slc file " + filename, c=1)
            return None
    elif ".vti" in filename.lower():
        reader = vtk.vtkXMLImageDataReader()
    elif ".mhd" in filename.lower():
        reader = vtk.vtkMetaImageReader()
    elif ".dem" in filename.lower():
        reader = vtk.vtkDEMReader()
    elif ".nii" in filename.lower():
        reader = vtk.vtkNIFTIImageReader()
    elif ".nrrd" in filename.lower():
        reader = vtk.vtkNrrdReader()
        if not reader.CanReadFile(filename):
            colors.printc("~prohibited Sorry bad nrrd file " + filename, c=1)
            return None
    reader.SetFileName(filename)
    reader.Update()
    image = reader.GetOutput()
    if len(spacing) == 3:
        image.SetSpacing(spacing[0], spacing[1], spacing[2])
    return image



###########################################################
def write(objct, fileoutput, binary=True):
    """
    Write 3D object to file. (same as `save()`).

    Possile extensions are:
        - vtk, vti, ply, obj, stl, byu, vtp, vti, mhd, xyz, tif, png, bmp.
    """
    obj = objct
    if isinstance(obj, Actor): # picks transformation
        obj = objct.polydata(True)
    elif isinstance(obj, (vtk.vtkActor, vtk.vtkVolume)):
        obj = objct.GetMapper().GetInput()
    elif isinstance(obj, (vtk.vtkPolyData, vtk.vtkImageData)):
        obj = objct

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
    elif ".tif" in fr:
        w = vtk.vtkTIFFWriter()
        w.SetFileDimensionality(len(obj.GetDimensions()))
    elif ".vti" in fr:
        w = vtk.vtkXMLImageDataWriter()
    elif ".mhd" in fr:
        w = vtk.vtkMetaImageWriter()
    elif ".png" in fr:
        w = vtk.vtkPNGWriter()
    elif ".jpg" in fr:
        w = vtk.vtkJPEGWriter()
    elif ".bmp" in fr:
        w = vtk.vtkBMPWriter()
    elif ".xml" in fr:  # write tetrahedral dolfin xml
        vertices = obj.coordinates()
        faces = obj.cells()
        ncoords = vertices.shape[0]
        ntets = faces.shape[0]
        outF = open(fileoutput, "w")
        outF.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        outF.write('<dolfin xmlns:dolfin="http://www.fenicsproject.org">\n')
        outF.write('  <mesh celltype="tetrahedron" dim="3">\n')
        outF.write('    <vertices size="' + str(ncoords) + '">\n')
        for i in range(ncoords):
            x, y, z = vertices[i]
            outF.write('      <vertex index="'+str(i)
                       + '" x="'+str(x)+'" y="'+str(y)+'" z="'+str(z)+'"/>\n')
        outF.write('    </vertices>\n')
        outF.write('    <cells size="' + str(ntets) + '">\n')
        for i in range(ntets):
            v0, v1, v2, v3 = faces[i]
            outF.write('      <tetrahedron index="'+str(i)
                       + '" v0="'+str(v0)+'" v1="'+str(v1)+'" v2="'+str(v2)+'" v3="'+str(v3)+'"/>\n')
        outF.write('    </cells>\n')
        outF.write("  </mesh>\n")
        outF.write("</dolfin>\n")
        outF.close()
        return objct
    else:
        colors.printc("~noentry Unknown format", fileoutput, "file not saved.", c="r")
        return objct

    try:
        if hasattr(w, 'SetFileTypeToBinary'):
            if binary:
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

def save(objct, fileoutput, binary=True):
    """
    Save 3D object to file. (same as `write()`).

    Possile extensions are:
        - vtk, vti, ply, obj, stl, vtp, xyz, tif, vti, mhd, png, bmp.
    """
    return write(objct, fileoutput, binary)


###########################################################
def exportWindow(fileoutput, binary=False, speed=None, html=True):
    '''
    Exporter which writes out the renderered scene into an OBJ or X3D file.
    X3D is an XML-based format for representation 3D scenes (similar to VRML).
    Check out http://www.web3d.org/x3d for more details.

    :param float speed: set speed for x3d files.
    :param bool html: generate a test html page for x3d files.

    |export_x3d| |export_x3d.py|_

        `generated webpage <https://vtkplotter.embl.es/examples/embryo.html>`_

        See also: FEniCS test `webpage <https://vtkplotter.embl.es/examples/fenics_elasticity.html>`_.
    '''
    fr = fileoutput.lower()
    if ".obj" in fr:
        w = vtk.vtkOBJExporter()
        w.SetInputData(settings.plotter_instance.window)
        w.Update()
        colors.printc("~save Saved file:", fileoutput, c="g")
    elif ".x3d" in fr:
        exporter = vtk.vtkX3DExporter()
        exporter.SetBinary(binary)
        exporter.FastestOff()
        if speed:
            exporter.SetSpeed(speed)
        exporter.SetInput(settings.plotter_instance.window)
        exporter.SetFileName(fileoutput)
        exporter.Update()
        exporter.Write()
        if not html:
            return
        from vtkplotter.docs import x3d_html
        x3d_html = x3d_html.replace("~fileoutput", fileoutput)
        wsize = settings.plotter_instance.window.GetSize()
        x3d_html = x3d_html.replace("~width", str(wsize[0]))
        x3d_html = x3d_html.replace("~height", str(wsize[1]))
        #b = settings.plotter_instance.renderer.ComputeVisiblePropBounds()
        #s = max(b[1] - b[0], b[3] - b[2], b[5] - b[4])
        #c = (b[1] + b[0])/2, (b[3] + b[2])/2, (b[5] + b[4])/2
        #x3d_html = x3d_html.replace("~size", str(s*2))
        #x3d_html = x3d_html.replace("~center", str(c[0])+" "+str(c[1])+" "+str(c[2]))
        outF = open(fileoutput.replace('.x3d', '.html'), "w")
        outF.write(x3d_html)
        outF.close()
        colors.printc("~save Saved files:", fileoutput,
                      fileoutput.replace('.x3d', '.html'), c="g")
    return


###########################################################
def buildPolyDataFast(vertices, faces=None, indexOffset=None):
    """
    Build a ``vtkPolyData`` object from a list of vertices
    where faces represents the connectivity of the polygonal mesh.

    E.g. :
        - ``vertices=[[x1,y1,z1],[x2,y2,z2], ...]``
        - ``faces=[[0,1,2], [1,2,3], ...]``
    """
    from vtk.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray

    dts = vtk.vtkIdTypeArray().GetDataTypeSize()
    ast = numpy.int32
    if dts != 4:
        ast = numpy.int64

    if not utils.isSequence(vertices):  # assume a dolfin.Mesh
        from dolfin import Mesh, BoundaryMesh
        mesh = Mesh(vertices)
        mesh = BoundaryMesh(mesh, "exterior")
        vertices = mesh.coordinates()
        faces = mesh.cells()

    # must fix dim=3 of vertices.. todo

    poly = vtk.vtkPolyData()
    vpts = vtk.vtkPoints()
    vpts.SetData(numpy_to_vtk(vertices, deep=True))
    poly.SetPoints(vpts)

    cells = vtk.vtkCellArray()
    if faces is not None:
        nf, nc = faces.shape
        dts = vtk.vtkIdTypeArray().GetDataTypeSize()
        ast = numpy.int32
        if dts != 4:
            ast = numpy.int64
        hs = numpy.hstack((numpy.zeros(nf)[:,None] + nc, faces)).astype(ast).ravel()
        arr = numpy_to_vtkIdTypeArray(hs, deep=True)
        cells.SetCells(nf, arr)
        poly.SetPolys(cells)
    else:
        sourceVertices = vtk.vtkCellArray()
        for i in range(len(vertices)):
            sourceVertices.InsertNextCell(1)
            sourceVertices.InsertCellPoint(i)
        poly.SetVerts(sourceVertices)

    return poly


def buildPolyData(vertices, faces=None, indexOffset=0):
    """
    Build a ``vtkPolyData`` object from a list of vertices
    where faces represents the connectivity of the polygonal mesh.

    E.g. :
        - ``vertices=[[x1,y1,z1],[x2,y2,z2], ...]``
        - ``faces=[[0,1,2], [1,2,3], ...]``

    Use ``indexOffset=1`` if face numbering starts from 1 instead of 0.
    """
    if 'dolfin' in str(vertices):  # assume a dolfin.Mesh
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
            pb = utils.ProgressBar(0, len(faces), ETA=False)
        for f in faces:
            n = len(f)
            if n == 4: #ugly but a bit faster:
                ele0 = vtk.vtkTriangle()
                ele1 = vtk.vtkTriangle()
                ele2 = vtk.vtkTriangle()
                ele3 = vtk.vtkTriangle()
                if indexOffset:
                    for i in [0,1,2,3]:
                        f[i] -= indexOffset
                f0, f1, f2, f3 = f
                pid0 = ele0.GetPointIds()
                pid1 = ele1.GetPointIds()
                pid2 = ele2.GetPointIds()
                pid3 = ele3.GetPointIds()

                pid0.SetId(0, f0)
                pid0.SetId(1, f1)
                pid0.SetId(2, f2)

                pid1.SetId(0, f0)
                pid1.SetId(1, f1)
                pid1.SetId(2, f3)

                pid2.SetId(0, f1)
                pid2.SetId(1, f2)
                pid2.SetId(2, f3)

                pid3.SetId(0, f2)
                pid3.SetId(1, f3)
                pid3.SetId(2, f0)

                sourcePolygons.InsertNextCell(ele0)
                sourcePolygons.InsertNextCell(ele1)
                sourcePolygons.InsertNextCell(ele2)
                sourcePolygons.InsertNextCell(ele3)

#            if n == 4: #problematic because of faces orientation
#                ele = vtk.vtkTetra()
#                pids = ele.GetPointIds()
#                for i in reversed(range(4)):
#                    pids.SetId(i, f[i] - indexOffset)
#                sourcePolygons.InsertNextCell(ele)

            elif n == 3:
                ele = vtk.vtkTriangle()
                pids = ele.GetPointIds()
                for i in range(3):
                    pids.SetId(i, f[i] - indexOffset)
                sourcePolygons.InsertNextCell(ele)

            else:
                ele = vtk.vtkPolygon()
                pids = ele.GetPointIds()
                pids.SetNumberOfIds(n)
                for i in range(n):
                    pids.SetId(i, f[i] - indexOffset)
                sourcePolygons.InsertNextCell(ele)
            if showbar:
                pb.print("converting mesh...    ")

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
    w2if.SetInput(settings.plotter_instance.window)
    s = settings.screeshotScale
    w2if.SetScale(s, s)
    if settings.screenshotTransparentBackground:
        w2if.SetInputBufferTypeToRGBA()
    w2if.ReadFrontBufferOff()  # read from the back buffer
    w2if.Update()
    pngwriter = vtk.vtkPNGWriter()
    pngwriter.SetFileName(filename)
    pngwriter.SetInputConnection(w2if.GetOutputPort())
    pngwriter.Write()


class Video:
    """
    Class to generate a video from the specified rendering window.
    Program ``ffmpeg`` is used to create video from each generated frame.
    :param str name: name of the output file.
    :param int fps: set the number of frames per second.
    :param float duration: set the total `duration` of the video and recalculates `fps` accordingly.
    :param str ffmpeg: set path to ffmpeg program. Default value considers ffmpeg is in the path.

    |makeVideo| |makeVideo.py|_
    """

    def __init__(self, name="movie.avi", **kwargs):

        from tempfile import TemporaryDirectory

        self.name = name
        self.duration = kwargs.pop('duration', None)
        self.fps = float(kwargs.pop('fps', 12))
        self.ffmpeg = kwargs.pop('ffmpeg', 'ffmpeg')
        self.frames = []
        self.tmp_dir = TemporaryDirectory()
        self.get_filename = lambda x: os.path.join(self.tmp_dir.name, x)
        colors.printc("~video Video", name, "is open...", c="m")

    def addFrame(self):
        """Add frame to current video."""
        fr = self.get_filename(str(len(self.frames)) + ".png")
        screenshot(fr)
        self.frames.append(fr)

    def pause(self, pause=0):
        """Insert a `pause`, in seconds."""
        fr = self.frames[-1]
        n = int(self.fps * pause)
        for _ in range(n):
            fr2 = self.get_filename(str(len(self.frames)) + ".png")
            self.frames.append(fr2)
            os.system("cp -f %s %s" % (fr, fr2))

    def close(self):
        """Render the video and write to file."""
        if self.duration:
            self.fps = len(self.frames) / float(self.duration)
            colors.printc("Recalculated video FPS to", round(self.fps, 3), c="yellow")
        else:
            self.fps = int(self.fps)
        self.name = self.name.split('.')[0]+'.mp4'
        out = os.system(self.ffmpeg + " -loglevel panic -y -r " + str(self.fps)
                        + " -i " + self.tmp_dir.name + os.sep + "%01d.png " + self.name)
        if out:
            colors.printc("ffmpeg returning error", c=1)
        colors.printc("~save Video saved as", self.name, c="green")
        self.tmp_dir.cleanup()
        return

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
    #print('Pressed key:', key, [vp])

    if key in ["q", "Q", "space", "Return"]:
        iren.ExitCallback()
        return

    elif key == "Escape":
        sys.stdout.flush()
        settings.plotter_instance.closeWindow()

    elif key in ["F1", "Pause"]:
        sys.stdout.flush()
        colors.printc('\n[F1] Execution aborted. Exiting python now.')
        settings.plotter_instance.closeWindow()
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

    if key == "C":
        cam = vp.renderer.GetActiveCamera()
        print('\n### Example code to position this vtkCamera:')
        print('vp = vtkplotter.Plotter()\n...')
        print('vp.camera.SetPosition(',   [round(e, 3) for e in cam.GetPosition()],  ')')
        print('vp.camera.SetFocalPoint(', [round(e, 3) for e in cam.GetFocalPoint()], ')')
        print('vp.camera.SetViewUp(',     [round(e, 3) for e in cam.GetViewUp()], ')')
        print('vp.camera.SetDistance(',   round(cam.GetDistance(), 3), ')')
        print('vp.camera.SetClippingRange(', [round(e, 3) for e in cam.GetClippingRange()], ')')
        return

    if key == "s":
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
        vp.icol += 1
        if vp.clickedActor and hasattr(vp.clickedActor, "GetProperty"):
            if (vp.icol) % 10 == 0:
                vp.clickedActor.GetMapper().ScalarVisibilityOn()
            else:
                vp.clickedActor.GetMapper().ScalarVisibilityOff()
                vp.clickedActor.GetProperty().SetColor(colors.colors1[(vp.icol) % 10])
        else:
            for i, ia in enumerate(vp.getActors()):
                if not ia.GetPickable():
                    continue
                ia.GetProperty().SetColor(colors.colors1[(i + vp.icol) % 10])
                ia.GetMapper().ScalarVisibilityOff()
        addons.addLegend()

    elif key == "2":
        vp.icol += 1
        if vp.clickedActor and hasattr(vp.clickedActor, "GetProperty"):
            if (vp.icol) % 10 == 0:
                vp.clickedActor.GetMapper().ScalarVisibilityOn()
            else:
                vp.clickedActor.GetMapper().ScalarVisibilityOff()
                vp.clickedActor.GetProperty().SetColor(colors.colors2[(vp.icol) % 10])
        else:
            for i, ia in enumerate(vp.getActors()):
                if not ia.GetPickable():
                    continue
                ia.GetProperty().SetColor(colors.colors2[(i + vp.icol) % 10])
                ia.GetMapper().ScalarVisibilityOff()
        addons.addLegend()

    elif key == "3":
        c = colors.getColor("gold")
        acs = vp.getActors()
        if len(acs) == 0: return
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
            if vp.axes_instances[clickedr]:
                if hasattr(vp.axes_instances[clickedr], "EnabledOff"):  # widget
                    vp.axes_instances[clickedr].EnabledOff()
                else:
                    vp.renderer.RemoveActor(vp.axes_instances[clickedr])
                vp.axes_instances[clickedr] = None
            addons.addAxes(axtype=asso[key], c=None)
            vp.interactor.Render()

    elif key in ["k", "K"]:
        for a in vp.getActors():
            ptdata = a.GetMapper().GetInput().GetPointData()
            cldata = a.GetMapper().GetInput().GetCellData()

            arrtypes = dict()
            arrtypes[vtk.VTK_UNSIGNED_CHAR] = "UNSIGNED_CHAR"
            arrtypes[vtk.VTK_UNSIGNED_INT] = "UNSIGNED_INT"
            arrtypes[vtk.VTK_FLOAT] = "FLOAT"
            arrtypes[vtk.VTK_DOUBLE] = "DOUBLE"
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

    elif key == "l":
        if vp.clickedActor in vp.getActors():
            acts = [vp.clickedActor]
        else:
            acts = vp.getActors()
        for ia in acts:
            if not ia.GetPickable():
                continue
            try:
                ev = ia.GetProperty().GetEdgeVisibility()
                ia.GetProperty().SetEdgeVisibility(not ev)
                ia.GetProperty().SetRepresentationToSurface()
                ia.GetProperty().SetLineWidth(0.1)
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
