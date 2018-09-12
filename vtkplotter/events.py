from __future__ import division, print_function
import sys
import vtk

import vtkplotter.utils as utils
import vtkplotter.colors as colors


############################### mouse event
def mouseleft(vp, obj, event):

    x,y = vp.interactor.GetEventPosition()
    #print ('mouse at',x,y)
    
    vp.renderer = obj.FindPokedRenderer(x,y)
    vp.renderWin = obj.GetRenderWindow()
    clickedr = vp.renderers.index(vp.renderer)
    picker = vtk.vtkPropPicker()
    picker.PickProp(x,y, vp.renderer)
    clickedActor = picker.GetActor()
    clickedActorIsAssembly = False
    if not clickedActor: 
        clickedActor = picker.GetAssembly()
        clickedActorIsAssembly = True
    vp.picked3d = picker.GetPickPosition()
    vp.justremoved = None

    if vp.verbose:
        if len(vp.renderers)>1 or clickedr>0 and vp.clickedr != clickedr:
            print ('Current Renderer:', clickedr, end='')
            print (', nr. of actors =', len(vp.getActors()))
        
        leg, oldleg = '', ''
        if hasattr(clickedActor,'legend'): leg = clickedActor.legend
        if hasattr(vp.clickedActor,'legend'): oldleg = vp.clickedActor.legend
        #detect if clickin the same obj
        if leg and isinstance(leg, str) and len(leg) and oldleg != leg: 
            try: indx = str(vp.getActors().index(clickedActor))
            except ValueError: indx = None                        
            try: indx = str(vp.actors.index(clickedActor))
            except ValueError: indx = None                        
            try: 
                rgb = list(clickedActor.GetProperty().GetColor())
                cn = colors.getColorName(rgb)
                if cn == 'white': cn = ''
                else: cn = '('+cn+'),'
            except: 
                cn = ''                        
            if indx and isinstance(clickedActor, vtk.vtkAssembly): 
                colors.printc('-> assembly',indx+':',clickedActor.legend,cn, end=' ')
            elif indx:
                colors.printc('-> actor', indx+':', leg, cn, end=' ')
            if not clickedActorIsAssembly:
                n = clickedActor.GetMapper().GetInput().GetNumberOfPoints()
            else:
                n = vp.getActors(clickedActor)[0].GetMapper().GetInput().GetNumberOfPoints()
            colors.printc('N='+str(n), end='')
            px,py,pz = vp.picked3d
            px,py,pz = str(round(px,1)), str(round(py,1)), str(round(pz,1))
            colors.printc(', p=('+px+','+py+','+pz+')')

    vp.clickedActor = clickedActor
    vp.clickedRenderer = clickedr


############################### keystroke event
def keypress(vp, obj, event):
    
    key = obj.GetKeySym()
    #print ('Pressed key:', key, event)

    if key == "q" or key == "space" or key == "Return":
        vp.interactor.ExitCallback()
        return

    elif key == "e":
        if vp.verbose: 
            print ("closing window...")
        vp.interactor.GetRenderWindow().Finalize()
        vp.interactor.TerminateApp()
        return

    elif key == "Escape":
        print()
        sys.stdout.flush()
        vp.interactor.TerminateApp()
        vp.interactor.GetRenderWindow().Finalize()
        vp.interactor.TerminateApp()
        sys.exit(0)
        
    elif key == "m":
        if vp.clickedActor in vp.getActors():
            vp.clickedActor.GetProperty().SetOpacity(0.05)
            bfp = vp.clickedActor.GetBackfaceProperty()
            if bfp: bfp.SetOpacity(0.05)
        else:
            for a in vp.getActors(): 
                a.GetProperty().SetOpacity(.05)
                bfp = a.GetBackfaceProperty()
                if bfp: bfp.SetOpacity(0.05)

    elif key == "slash":
        if vp.clickedActor in vp.getActors():
            vp.clickedActor.GetProperty().SetOpacity(1) 
            bfp = vp.clickedActor.GetBackfaceProperty()
            if bfp: bfp.SetOpacity(1)
        else:
            for a in vp.getActors(): 
                a.GetProperty().SetOpacity(1)
                bfp = a.GetBackfaceProperty()
                if bfp: bfp.SetOpacity(1)

    elif key == "comma":
        if vp.clickedActor in vp.getActors():
            ap = vp.clickedActor.GetProperty()
            ap.SetOpacity(max([ap.GetOpacity()-0.05, 0.05]))
            bfp = vp.clickedActor.GetBackfaceProperty()
            if bfp: bfp.SetOpacity(ap.GetOpacity())
        else:
            for a in vp.getActors():
                ap = a.GetProperty()
                ap.SetOpacity(max([ap.GetOpacity()-0.05, 0.05]))
                bfp = a.GetBackfaceProperty()
                if bfp: bfp.SetOpacity(ap.GetOpacity())

    elif key == "period":
        if vp.clickedActor in vp.getActors():
            ap = vp.clickedActor.GetProperty()
            ap.SetOpacity(min([ap.GetOpacity()+0.05, 1.0]))
            bfp = vp.clickedActor.GetBackfaceProperty()
            if bfp: bfp.SetOpacity(ap.GetOpacity())
        else:
            for a in vp.getActors():
                ap = a.GetProperty()
                ap.SetOpacity(min([ap.GetOpacity()+0.05, 1.0]))
                bfp = a.GetBackfaceProperty()
                if bfp: bfp.SetOpacity(ap.GetOpacity())

    elif key == "P":
        if vp.clickedActor in vp.getActors(): acts=[vp.clickedActor]
        else: acts = vp.getActors()
        for ia in acts:
            try:
                ps = ia.GetProperty().GetPointSize()
                if ps>1:
                    ia.GetProperty().SetPointSize(ps-1)
                ia.GetProperty().SetRepresentationToPoints()
            except AttributeError: pass

    elif key == "p":
        if vp.clickedActor in vp.getActors(): acts=[vp.clickedActor]
        else: acts = vp.getActors()
        for ia in acts:
            try:
                ps = ia.GetProperty().GetPointSize()
                ia.GetProperty().SetPointSize(ps+2)
                ia.GetProperty().SetRepresentationToPoints()
            except AttributeError: pass

    elif key == "w":
        if vp.clickedActor and vp.clickedActor in vp.getActors():
            vp.clickedActor.GetProperty().SetRepresentationToWireframe()
        else:
            for a in vp.getActors(): 
                if a: 
                    a.GetProperty().SetRepresentationToWireframe()

    elif key == "r":
        vp.renderer.ResetCamera()


    ### now intercept custom observer ###########################
    if vp.keyPressFunction: 
        if key not in ['Shift_L', 'Control_L', 'Super_L', 'Alt_L']:
            if key not in ['Shift_R', 'Control_R', 'Super_R', 'Alt_R']:
                vp.verbose = False
                vp.keyPressFunction(key, vp)
                return


    if key == "S":
        colors.printc('Saving window as screenshot.png', c='green')
        vp.screenshot('screenshot.png')
        return

    elif key == "C":
        cam = vp.renderer.GetActiveCamera()
        print ('\n### Example code to position this vtkCamera:')
        print ('vp = vtkplotter.Plotter()\n...')
        print ('vp.camera.SetPosition(',  [round(e,3) for e in cam.GetPosition()],  ')')
        print ('vp.camera.SetFocalPoint(',[round(e,3) for e in cam.GetFocalPoint()],')')
        print ('vp.camera.SetViewUp(',    [round(e,3) for e in cam.GetViewUp()],')')
        print ('vp.camera.SetDistance(',   round(cam.GetDistance() ,3),')')
        print ('vp.camera.SetClippingRange(', [round(e,3) for e in cam.GetClippingRange()],')')
        return

    elif key == "s":
        if vp.clickedActor and vp.clickedActor in vp.getActors():
            vp.clickedActor.GetProperty().SetRepresentationToSurface()
        else:
            for a in vp.getActors(): 
                if a: 
                    a.GetProperty().SetRepresentationToSurface()

    elif key == "V":
        if not(vp.verbose): vp._tips()
        vp.verbose = not(vp.verbose)
        print ("Verbose: ", vp.verbose)

    elif key in ["1", "KP_End", "KP_1"]:
        for i,ia in enumerate(vp.getActors()):
            ia.GetProperty().SetColor(colors.colors1[(i+vp.icol)%10])
            ia.GetMapper().ScalarVisibilityOff()
        vp.icol += 1
        vp._draw_legend()

    elif key in ["2", "KP_Down", "KP_2"]:
        for i,ia in enumerate(vp.getActors()):
            ia.GetProperty().SetColor(colors.colors2[(i+vp.icol)%10])
            ia.GetMapper().ScalarVisibilityOff()
        vp.icol += 1
        vp._draw_legend()

    elif key in ["3", "KP_Left", "KP_4"]:
        for i,ia in enumerate(vp.getActors()):
            ia.GetProperty().SetColor(colors.colors3[(i+vp.icol)%10])
            ia.GetMapper().ScalarVisibilityOff()
        vp.icol += 1
        vp._draw_legend()

    elif key in ["4", "KP_Begin", "KP_5"]:
        c = colors.getColor('gold')
        acs = vp.getActors()
        alpha = 1./len(acs)
        for ia in acs:
            ia.GetProperty().SetColor(c)
            ia.GetProperty().SetOpacity(alpha)
            ia.GetMapper().ScalarVisibilityOff()
        vp._draw_legend()


    elif key in ["k", "K"]:
        for a in vp.getActors():
            ptdata = utils.polydata(a).GetPointData()
            cldata = utils.polydata(a).GetCellData()

            arrtypes = dict()
            arrtypes[vtk.VTK_UNSIGNED_CHAR] = 'VTK_UNSIGNED_CHAR'
            arrtypes[vtk.VTK_UNSIGNED_INT] = 'VTK_UNSIGNED_INT'
            arrtypes[vtk.VTK_FLOAT] = 'VTK_FLOAT'
            arrtypes[vtk.VTK_DOUBLE] = 'VTK_DOUBLE'
            foundarr=0

            if key == 'k':
                for i in range(ptdata.GetNumberOfArrays()):
                    name = ptdata.GetArrayName(i)
                    if name == 'Normals': continue
                    if vp.verbose:
                        if hasattr(a, 'legend'): print ('actor:', a.legend, end='')
                        print ('\tfound POINT array with name:', name, end=', ')
                        try: 
                            print ('type:', arrtypes[ptdata.GetArray(i).GetDataType()])
                        except:
                            print ('type:', ptdata.GetArray(i).GetDataType())
                    ptdata.SetActiveScalars(name)
                    foundarr=1
                if not foundarr:
                    print('No vtkArray is associated to points', end='')
                    if hasattr(a, 'legend'):
                        print(' for actor:', a.legend)
                    else: print()

            if key == 'K':
                for i in range(cldata.GetNumberOfArrays()):
                    name = cldata.GetArrayName(i)
                    if name == 'Normals': continue
                    if vp.verbose:
                        if hasattr(a, 'legend'): print ('actor:', a.legend, end='')
                        print ('\tfound POINT array', 'tname:', name, end=', ')
                        try:
                            print ('type:', arrtypes[cldata.GetArray(i).GetDataType()])
                        except:
                            print ('type:', cldata.GetArray(i).GetDataType())
                    cldata.SetActiveScalars(name)
                    foundarr=1
                if not foundarr:
                    print('No vtkArray is associated to cells', end='')
                    if hasattr(a, 'legend'):
                        print(' for actor:', a.legend)
                    else: print()
                    
            a.GetMapper().ScalarVisibilityOn()

    elif key == "L":
        if vp.clickedActor in vp.getActors(): acts=[vp.clickedActor]
        else: acts = vp.getActors()
        for ia in acts:
            try:
                ia.GetProperty().SetRepresentationToSurface()
                ls = ia.GetProperty().GetLineWidth()
                if ls<=1: 
                    ls=1
                    ia.GetProperty().EdgeVisibilityOff()
                else: 
                    ia.GetProperty().EdgeVisibilityOn()
                    ia.GetProperty().SetLineWidth(ls-1)
            except AttributeError: pass

    elif key == "l":
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
        if vp.clickedActor in vp.getActors(): 
            acts=[vp.clickedActor]
        else: 
            acts = vp.getActors()
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
            vp.render(a)

    elif key == "x":
        if vp.justremoved is None:                    
            if vp.clickedActor in vp.getActors() or isinstance(vp.clickedActor, vtk.vtkAssembly):
                vp.justremoved = vp.clickedActor
                vp.renderer.RemoveActor(vp.clickedActor)
            if hasattr(vp.clickedActor, 'legend') and vp.clickedActor.legend:
                colors.printc('   ...removing actor: '+ str(vp.clickedActor.legend)+', press x to put it back')
            else: 
                if vp.verbose:
                    colors.printc('Click an actor and press x to toggle it.',c=5)
        else:
            vp.renderer.AddActor(vp.justremoved)
            vp.renderer.Render()
            vp.justremoved = None
        vp._draw_legend()

    elif key == "X":
        if vp.clickedActor:
            if hasattr(vp.clickedActor, 'legend') and vp.clickedActor.legend:
                fname = 'clipped_'+str(vp.clickedActor.legend)
                fname = fname.split('.')[0]+'.vtk'
            else: fname = 'clipped.vtk'
            if vp.verbose:
                colors.printc('Move handles to remove part of the actor.',c=4)
            utils.cutterWidget(vp.clickedActor, fname) 
        elif vp.verbose: 
            colors.printc('Click an actor and press X to open the cutter box widget.',c=4)


    if vp.interactor: 
        vp.interactor.Render()




