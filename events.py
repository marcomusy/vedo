# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 11:15:37 2017

@author: mmusy
"""
from __future__ import division, print_function
import vtk
from vtkutils import *
from colors import getColorName



############################### mouse event
def _mouseleft(vp, obj, event):

    x,y = vp.interactor.GetEventPosition()
    #print ('mouse at',x,y)
    
    vp.renderer = obj.FindPokedRenderer(x,y)
    vp.renderWin = obj.GetRenderWindow()
    clickedr = vp.renderers.index(vp.renderer)
    picker = vtk.vtkPropPicker()
    picker.PickProp(x,y, vp.renderer)
    clickedActor = picker.GetActor()
    if not clickedActor: 
        clickedActor = picker.GetAssembly()
        
    if vp.verbose:
        if len(vp.renderers)>1 or clickedr>0 and vp.clickedr != clickedr:
            print ('Current Renderer:', clickedr, end='')
            print (', nr. of actors =', len(vp.getActors()))
        
        leg, oldleg = '', ''
        if hasattr(clickedActor,'legend'): leg = clickedActor.legend
        if hasattr(vp.clickedActor,'legend'): oldleg = vp.clickedActor.legend
        if leg and len(leg) and oldleg != leg: #detect if clickin the same obj
            try: indx = str(vp.getActors().index(clickedActor))
            except ValueError: indx = None                        
            try: indx = str(vp.actors.index(clickedActor))
            except ValueError: indx = None                        
            try: 
                rgb = list(clickedActor.GetProperty().GetColor())
                cn = '('+getColorName(rgb)+'),'
            except: 
                cn = None                        
            if indx and isinstance(clickedActor, vtk.vtkAssembly): 
                printc(('-> assembly', indx+':', clickedActor.legend), end=' ')
            elif indx:
                printc(('-> actor', indx+':', leg), end=' ')
                if cn: printc(cn, c=0, end=' ')
            printc('N='+str(getPolyData(clickedActor).GetNumberOfPoints()))
                
    vp.clickedActor = clickedActor
    vp.clickedr = clickedr


############################### keystroke event
def _keypress(vp, obj, event):
    
    key = obj.GetKeySym()
    #print ('Pressed key:', key, event)

    if   key == "q" or key == "space" or key == "Return":
        vp.interactor.ExitCallback()

    elif key == "e":
        if vp.verbose: print ("closing window...")
        vp.interactor.GetRenderWindow().Finalize()
        vp.interactor.TerminateApp()
        del vp.renderWin, vp.interactor
        return

    elif key == "Escape":
        vp.interactor.TerminateApp()
        vp.interactor.GetRenderWindow().Finalize()
        vp.interactor.TerminateApp()
        del vp.renderWin, vp.interactor
        exit(0)

    elif key == "S":
        printc('Saving window as screenshot.png', 'green')
        screenshot()
        return

    elif key == "C":
        cam = vp.renderer.GetActiveCamera()
        print ('\nfrom vtk import vtkCamera ### example code')
        print ('cam = vtkCamera()')
        print ('cam.SetPosition(',  [round(e,3) for e in cam.GetPosition()],  ')')
        print ('cam.SetFocalPoint(',[round(e,3) for e in cam.GetFocalPoint()],')')
        print ('cam.SetParallelScale(',round(cam.GetParallelScale(),3),')')
        print ('cam.SetViewUp(', [round(e,3) for e in cam.GetViewUp()],')\n')
        return

    elif key == "m":
        if vp.clickedActor in vp.getActors():
            vp.clickedActor.GetProperty().SetOpacity(0.05)
        else:
            for a in vp.getActors(): a.GetProperty().SetOpacity(.05)

    elif key == "comma":
        if vp.clickedActor in vp.getActors():
            ap = vp.clickedActor.GetProperty()
            ap.SetOpacity(max([ap.GetOpacity()-0.05, 0.05]))
        else:
            for a in vp.getActors():
                ap = a.GetProperty()
                ap.SetOpacity(max([ap.GetOpacity()-0.05, 0.05]))

    elif key == "period":
        if vp.clickedActor in vp.getActors():
            ap = vp.clickedActor.GetProperty()
            ap.SetOpacity(min([ap.GetOpacity()+0.05, 1.0]))
        else:
            for a in vp.getActors():
                ap = a.GetProperty()
                ap.SetOpacity(min([ap.GetOpacity()+0.05, 1.0]))

    elif key == "slash":
        if vp.clickedActor in vp.getActors():
            vp.clickedActor.GetProperty().SetOpacity(1) 
        else:
            for a in vp.getActors(): a.GetProperty().SetOpacity(1)

    elif key == "V":
        if not(vp.verbose): vp._tips()
        vp.verbose = not(vp.verbose)
        print ("Verbose: ", vp.verbose)

    elif key in ["1", "KP_End", "KP_1"]:
        for i,ia in enumerate(vp.getActors()):
            ia.GetProperty().SetColor(colors1[i+vp.icol1])
        vp.icol1 += 1
        vp._draw_legend()

    elif key in ["2", "KP_Down", "KP_2"]:
        for i,ia in enumerate(vp.getActors()):
            ia.GetProperty().SetColor(colors2[i+vp.icol2])
        vp.icol2 += 1
        vp._draw_legend()

    elif key in ["4", "KP_Left", "KP_4"]:
        for i,ia in enumerate(vp.getActors()):
            ia.GetProperty().SetColor(colors3[i+vp.icol3])
        vp.icol3 += 1
        vp._draw_legend()

    elif key in ["5", "KP_Begin", "KP_5"]:
        c = getColor('gold')
        acs = vp.getActors()
        alpha = 1./len(acs)
        for ia in acs:
            ia.GetProperty().SetColor(c)
            ia.GetProperty().SetOpacity(alpha)
        vp._draw_legend()

    elif key == "o":
        if vp.clickedActor in vp.getActors(): acts=[vp.clickedActor]
        else: acts = vp.getActors()
        for ia in acts:
            try:
                ps = ia.GetProperty().GetPointSize()
                ia.GetProperty().SetPointSize(ps-1)
                ia.GetProperty().SetRepresentationToPoints()
            except AttributeError: pass

    elif key == "O":
        if vp.clickedActor in vp.getActors(): acts=[vp.clickedActor]
        else: acts = vp.getActors()
        for ia in acts:
            try:
                ps = ia.GetProperty().GetPointSize()
                ia.GetProperty().SetPointSize(ps+2)
                ia.GetProperty().SetRepresentationToPoints()
            except AttributeError: pass

    elif key == "l":
        if vp.clickedActor in vp.getActors(): acts=[vp.clickedActor]
        else: acts = vp.getActors()
        for ia in acts:
            try:
                ia.GetProperty().SetRepresentationToSurface()
                ls = ia.GetProperty().GetLineWidth()
                if ls==1: 
                    ia.GetProperty().EdgeVisibilityOff() 
                    ia.GetProperty().SetLineWidth(0)
                else: ia.GetProperty().SetLineWidth(ls-1)
            except AttributeError: pass

    elif key == "L":
        if vp.clickedActor in vp.getActors(): acts=[vp.clickedActor]
        else: acts = vp.getActors()
        for ia in acts:
            try:
                ia.GetProperty().EdgeVisibilityOn()
                c = ia.GetProperty().GetColor()
                ia.GetProperty().SetEdgeColor(c)
                ls = ia.GetProperty().GetLineWidth()
                ia.GetProperty().SetLineWidth(ls+1)
            except AttributeError: pass

    elif key == "n": # show normals to an actor
        if vp.clickedActor in vp.getActors(): acts=[vp.clickedActor]
        else: acts = vp.getActors()
        for ia in acts:
            alpha = ia.GetProperty().GetOpacity()
            c = ia.GetProperty().GetColor()
            a = vp.normals(ia, ratio=1, c=c, alpha=alpha)
            vp.actors.pop() #remove from list
            try:
                i = vp.actors.index(ia)
                vp.actors[i] = a
                vp.renderer.RemoveActor(ia)
                vp.interactor.Render()
            except ValueError: pass
        ii = bool(vp.interactive)
        vp.show(at=vp.clickedr, interactive=0, axes=0)
        vp.interactive = ii # restore it

    elif key == "x":
        vp.justremoved = None # needs fix
        if vp.justremoved is None:                    
            if isinstance(vp.clickedActor, vtk.vtkAssembly):
                props = vtk.vtkPropCollection()
                vp.clickedActor.GetActors(props)
                actr = props.GetLastProp()
                try:
                    al = np.sqrt(actr.GetProperty().GetOpacity())
                    for op in np.linspace(al,0, 8): #fade away
                        actr.GetProperty().SetOpacity(op)
                        vp.interactor.Render()
                except AttributeError: pass
                vp.justremoved = actr
                vp.clickedActor.RemovePart(actr)                    
            elif vp.clickedActor in vp.getActors():
                actr = vp.clickedActor
                al = np.sqrt(actr.GetProperty().GetOpacity())
                for op in np.linspace(al,0, 8): #fade away
                    actr.GetProperty().SetOpacity(op)
                    vp.interactor.Render()
                vp.justremoved = actr
                vp.renderer.RemoveActor(actr)
            else: 
                if vp.verbose:
                    printc('Click an actor and press x to remove it.',5)
                return
            if vp.verbose and hasattr(actr, 'legend'):
                print ('   ...removing actor:', actr.legend)
            vp._draw_legend()
        else:
            if isinstance(vp.clickedActor, vtk.vtkAssembly):
                vp.clickedActor.AddPart(vp.justremoved)
                vp._draw_legend()
            elif vp.clickedActor in vp.actors:
                print ([vp.clickedActor, vp.justremoved])
                vp.renderer.AddActor(vp.justremoved)
                vp.renderer.Render()
                vp._draw_legend()        
            vp.justremoved = None

    elif key == "X":
        if vp.clickedActor:
            if hasattr(vp.clickedActor, 'legend'):
                fname = 'clipped_'+vp.clickedActor.legend
                fname = fname.split('.')[0]+'.vtk'
            else: fname = 'clipped.vtk'
            if vp.verbose:
                printc('Move handles to remove part of the actor.',4)
            cutterWidget(vp.clickedActor, fname) 
        elif vp.verbose: 
            printc('Click an actor and press X to open the cutter box widget.',4)
        
    vp.interactor.Render()


############################### timer event
# allows to move the window while running
# see lines 1306, 1330 in plotter.py
def _stopren(vp, obj, event):
    #if vp.interactive: return
    #x,y = vp.interactor.GetEventPosition()
    #print (' _stopren at',x,y, event, obj.GetKeySym())
    vp.interactor.ExitCallback()
    
    


