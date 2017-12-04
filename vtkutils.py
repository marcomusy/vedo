# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 20:10:27 2017

@author: mmusy
"""
import vtk

#########################################################
# Useful Functions
######################################################### 
def screenshot(filename='screenshot.png'):
    try:
        import gtk.gdk
        w = gtk.gdk.get_default_root_window().get_screen().get_active_window()
        sz = w.get_size()
        pb = gtk.gdk.Pixbuf(gtk.gdk.COLORSPACE_RGB, False, 8, sz[0], sz[1])
        pb = pb.get_from_drawable(w,w.get_colormap(),0,0,0,0, sz[0], sz[1])
        if pb is not None:
            pb.save(filename, "png")
            #print ("Screenshot saved to", filename)
        else: print ("Unable to save the screenshot. Skip.")
    except:
        print ("Unable to take the screenshot. Skip.")


def makePolyData(spoints, addLines=True):
    """Try to workout a polydata from points"""
    sourcePoints = vtk.vtkPoints()
    sourceVertices = vtk.vtkCellArray()
    for pt in spoints:
        if len(pt)==3: #it's 3D!
            aid = sourcePoints.InsertNextPoint(pt[0], pt[1], pt[2])
        else:
            aid = sourcePoints.InsertNextPoint(pt[0], pt[1], 0)
        sourceVertices.InsertNextCell(1)
        sourceVertices.InsertCellPoint(aid)
    source = vtk.vtkPolyData()
    source.SetPoints(sourcePoints)
    source.SetVerts(sourceVertices)
    if addLines:
        lines = vtk.vtkCellArray()
        lines.InsertNextCell(len(spoints))
        for i in range(len(spoints)): lines.InsertCellPoint(i)
        source.SetLines(lines)
    source.Update()
    return source


def isInside(poly, point):
    """Return True if point is inside a polydata closed surface"""
    points = vtk.vtkPoints()
    points.InsertNextPoint(point)
    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(points)
    selectEnclosedPoints = vtk.vtkSelectEnclosedPoints()
    selectEnclosedPoints.SetInput(pointsPolydata)
    selectEnclosedPoints.SetSurface(poly)
    selectEnclosedPoints.Update()
    return selectEnclosedPoints.IsInside(0)
    

####################################
def write(poly, fileoutput):
    wt = vtk.vtkPolyDataWriter()
    setInput(wt, poly)
    wt.SetFileName(fileoutput)
    print ("Writing vtk file:", fileoutput)
    wt.Write()
    
vtkMV = vtk.vtkVersion().GetVTKMajorVersion() > 5
def setInput(vtkobj, p):
        if vtkMV: vtkobj.SetInputData(p)
        else: vtkobj.SetInput(p)
