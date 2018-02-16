#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 15:06:48 2017

@author: mmusy
"""
from __future__ import division, print_function
from glob import glob
import os
import vtkcolors
import vtk
import vtkutils as ut



####################################### LOADER
def load(filesOrDirs, c='gold', alpha=0.2, 
          wire=False, bc=None, edges=False, legend=True, texture=None):
    '''Returns a vtkActor from reading a file or directory. 
       Optional args:
       c,     color in RGB format, hex, symbol or name
       alpha, transparency (0=invisible)
       wire,  show surface as wireframe
       bc,    backface color of internal surface
       legend, text to show on legend, if True picks filename.
    '''
    acts = []
    if isinstance(legend, int): legend = bool(legend)
    if isinstance(filesOrDirs, list):
        flist = filesOrDirs
    else:
        flist = sorted(glob(filesOrDirs))
    for fod in flist:
        if os.path.isfile(fod): 
            a = _loadFile(fod, c, alpha, wire, bc, edges, legend, texture)
            acts.append(a)
        elif os.path.isdir(fod):
            acts = _loadDir(fod, c, alpha, wire, bc, edges, legend, texture)
    if not len(acts):
        ut.printc(('Cannot find:', filesOrDirs), c=1)
        exit(0) 
    if len(acts) == 1: return acts[0]
    else: return acts


def _loadFile(filename, c, alpha, wire, bc, edges, legend, texture):
    fl = filename.lower()
    if '.xml' in fl or '.xml.gz' in fl: # Fenics tetrahedral mesh file
        actor = _loadXml(filename, c, alpha, wire, bc, edges, legend)
    elif '.pcd' in fl:                  # PCL point-cloud format
        actor = _loadPCD(filename, c, alpha, legend)
    else:
        poly = _loadPoly(filename)
        if not poly:
            ut.printc(('Unable to load', filename), c=1)
            return False
        if legend is True: legend = os.path.basename(filename)
        actor = ut.makeActor(poly, c, alpha, wire, bc, edges, legend, texture)
        if '.txt' in fl or '.xyz' in fl: 
            actor.GetProperty().SetPointSize(4)
    return actor
    
def _loadDir(mydir, c, alpha, wire, bc, edges, legend, texture):
    if not os.path.exists(mydir): 
        ut.printc(('Error in loadDir: Cannot find', mydir), c=1)
        exit(0)
    acts = []
    for ifile in sorted(os.listdir(mydir)):
        _loadFile(mydir+'/'+ifile, c, alpha, wire, bc, edges, legend, texture)
    return acts

def _loadPoly(filename):
    '''Return a vtkPolyData object, NOT a vtkActor'''
    if not os.path.exists(filename): 
        ut.printc(('Error in loadPoly: Cannot find', filename), c=1)
        exit(0)
    fl = filename.lower()
    if   '.vtk' in fl: reader = vtk.vtkPolyDataReader()
    elif '.ply' in fl: reader = vtk.vtkPLYReader()
    elif '.obj' in fl: reader = vtk.vtkOBJReader()
    elif '.stl' in fl: reader = vtk.vtkSTLReader()
    elif '.byu' in fl or '.g' in fl: reader = vtk.vtkBYUReader()
    elif '.vtp' in fl: reader = vtk.vtkXMLPolyDataReader()
    elif '.vts' in fl: reader = vtk.vtkXMLStructuredGridReader()
    elif '.vtu' in fl: reader = vtk.vtkXMLUnstructuredGridReader()
    elif '.txt' in fl: reader = vtk.vtkParticleReader() # (x y z scalar) 
    elif '.xyz' in fl: reader = vtk.vtkParticleReader()
    else: reader = vtk.vtkDataReader()
    reader.SetFileName(filename)
    reader.Update()
    if '.vts' in fl: # structured grid
        gf = vtk.vtkStructuredGridGeometryFilter()
        gf.SetInputConnection(reader.GetOutputPort())
        gf.Update()
        poly = gf.GetOutput()
    elif '.vtu' in fl: # unstructured grid
        gf = vtk.vtkGeometryFilter()
        gf.SetInputConnection(reader.GetOutputPort())
        gf.Update()    
        poly = gf.GetOutput()
    else: poly = reader.GetOutput()
    
    if not poly: 
        ut.printc(('Unable to load', filename), c=1)
        return False
    
    mergeTriangles = vtk.vtkTriangleFilter()
    ut.setInput(mergeTriangles, poly)
    mergeTriangles.Update()
    poly = mergeTriangles.GetOutput()
    return poly


def _loadXml(filename, c, alpha, wire, bc, edges, legend):
    '''Reads a Fenics/Dolfin file format'''
    if not os.path.exists(filename): 
        ut.printc(('Error in loadXml: Cannot find', filename), c=1)
        exit(0)
    import xml.etree.ElementTree as et
    if '.gz' in filename:
        import gzip
        inF = gzip.open(filename, 'rb')
        outF = open('/tmp/filename.xml', 'wb')
        outF.write( inF.read() )
        outF.close()
        inF.close()
        tree = et.parse('/tmp/filename.xml')
    else: tree = et.parse(filename)
    coords, connectivity = [], []
    print('..loading',filename)
    for mesh in tree.getroot():
        for elem in mesh:
            for e in elem.findall('vertex'):
                x = float(e.get('x'))
                y = float(e.get('y'))
                z = float(e.get('z'))
                coords.append([x,y,z])
            for e in elem.findall('tetrahedron'):
                v0 = int(e.get('v0'))
                v1 = int(e.get('v1'))
                v2 = int(e.get('v2'))
                v3 = int(e.get('v3'))
                connectivity.append([v0,v1,v2,v3])
    points = vtk.vtkPoints()
    for p in coords: points.InsertNextPoint(p)

    ugrid = vtk.vtkUnstructuredGrid()
    ugrid.SetPoints(points)
    cellArray = vtk.vtkCellArray()
    for itet in range(len(connectivity)):
        tetra = vtk.vtkTetra()
        for k,j in enumerate(connectivity[itet]):
            tetra.GetPointIds().SetId(k, j)
        cellArray.InsertNextCell(tetra)
    ugrid.SetCells(vtk.VTK_TETRA, cellArray)

    # 3D cells are mapped only if they are used by only one cell,
    #  i.e., on the boundary of the data set
    mapper = vtk.vtkDataSetMapper()
    if ut.vtkMV: 
        mapper.SetInputData(ugrid)
    else:
        mapper.SetInputConnection(ugrid.GetProducerPort())

    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetInterpolationToFlat()
    actor.GetProperty().SetColor(vtkcolors.getColor(c))
    actor.GetProperty().SetOpacity(alpha/2.)
    #actor.GetProperty().VertexVisibilityOn()
    if edges: actor.GetProperty().EdgeVisibilityOn()
    if wire:  actor.GetProperty().SetRepresentationToWireframe()
    vpts = vtk.vtkPointSource()
    vpts.SetNumberOfPoints(len(coords))
    vpts.Update()
    vpts.GetOutput().SetPoints(points)
    pts_act = ut.makeActor(vpts.GetOutput(), c='b', alpha=alpha)
    pts_act.GetProperty().SetPointSize(3)
    pts_act.GetProperty().SetRepresentationToPoints()
    actor2 = ut.makeAssembly([pts_act, actor])
    if legend: setattr(actor2, 'legend', legend)
    if legend is True: 
        setattr(actor2, 'legend', os.path.basename(filename))
    return actor2
 

def _loadPCD(filename, c, alpha, legend):
    '''Return vtkActor from Point Cloud file format'''            
    if not os.path.exists(filename): 
        ut.printc(('Error in loadPCD: Cannot find file', filename), c=1)
        exit(0)
    f = open(filename, 'r')
    lines = f.readlines()
    f.close()
    start = False
    pts = []
    N, expN = 0, 0
    for text in lines:
        if start:
            if N >= expN: break
            l = text.split()
            pts.append([float(l[0]),float(l[1]),float(l[2])])
            N += 1
        if not start and 'POINTS' in text:
            expN= int(text.split()[1])
        if not start and 'DATA ascii' in text:
            start = True
    if expN != N:
        ut.printc(('Mismatch in pcd file', expN, len(pts)), 'red')
    src = vtk.vtkPointSource()
    src.SetNumberOfPoints(len(pts))
    src.Update()
    poly = src.GetOutput()
    for i,p in enumerate(pts): poly.GetPoints().SetPoint(i, p)
    if not poly:
        ut.printc(('Unable to load', filename), 'red')
        return False
    actor = ut.makeActor(poly, vtkcolors.getColor(c), alpha)
    actor.GetProperty().SetPointSize(4)
    if legend: setattr(actor, 'legend', legend)
    if legend is True: setattr(actor, 'legend', os.path.basename(filename))
    return actor
