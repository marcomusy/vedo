#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command Line Interface module
-----------------------------

    # Type for help
    vedo -h

    # Some useful bash aliases:
    alias v='vedo '
    alias vr='vedo --run '        # to search and run examples by name
    alias vs='vedo -i --search '  # to search for a string in examples
    alias ve='vedo --eog '        # to view single and multiple images (press h for help)
    alias vv='vedo -bg blackboard -bg2 gray3 -z 1.05 -k glossy -c blue9 '
"""
import argparse
import glob
import os
import sys

import numpy as np
import vedo
import vedo.applications as applications
import vtk
from vedo import __version__
from vedo import io
from vedo import load
from vedo import settings
from vedo.colors import getColor
from vedo.colors import printc
from vedo.mesh import Mesh
from vedo.picture import Picture
from vedo.plotter import Plotter
from vedo.tetmesh import TetMesh
from vedo.ugrid import UGrid
from vedo.utils import humansort
from vedo.utils import isSequence
from vedo.utils import printInfo
from vedo.volume import Volume


__all__ = []

##############################################################################################
def execute_cli():

    parser = get_parser()
    args = parser.parse_args()

    if "/vedo/vedo" in vedo.installdir:
        vedo.installdir = vedo.installdir.replace('vedo/','').replace('vedo\\','')

    if args.info is not None:
        system_info()

    elif args.run:
        exe_run(args)

    elif args.search:
        exe_search(args)

    elif args.search_vtk:
        exe_search_vtk(args)

    elif args.convert:
        exe_convert(args)

    elif args.eog:
        exe_eog(args)

    elif (len(args.files) == 0 or os.name == "nt"):
        exe_gui(args)

    else:
        draw_scene(args)


##############################################################################################
def get_parser():

    descr = f"version {__version__}"
    descr+= " - check out home page at https://vedo.embl.es"

    pr = argparse.ArgumentParser(description=descr)
    pr.add_argument('files', nargs='*',             help="input filename(s)")
    pr.add_argument("-c", "--color", type=str,      help="mesh color [integer or color name]", default=None, metavar='')
    pr.add_argument("-a", "--alpha",    type=float, help="alpha value [0-1]", default=1, metavar='')
    pr.add_argument("-w", "--wireframe",            help="use wireframe representation", action="store_true")
    pr.add_argument("-p", "--point-size", type=float, help="specify point size", default=-1, metavar='')
    pr.add_argument("-l", "--showedges",            help="show a thin line on mesh edges", action="store_true")
    pr.add_argument("-k", "--lighting", type=str,   help="metallic, plastic, shiny or glossy", default='default', metavar='')
    pr.add_argument("-K", "--flat",                 help="use flat shading", action="store_true")
    pr.add_argument("-t", "--texture-file",         help="texture image file", default='', metavar='')
    pr.add_argument("-x", "--axes-type", type=int,  help="specify axes type [0-14]", default=1, metavar='')
    pr.add_argument("-i", "--no-camera-share",      help="do not share camera in renderers", action="store_true")
    pr.add_argument("-f", "--full-screen",          help="full screen mode", action="store_true")
    pr.add_argument("-bg","--background", type=str, help="background color [integer or color name]", default='', metavar='')
    pr.add_argument("-bg2", "--background-grad",    help="use background color gradient", default='', metavar='')
    pr.add_argument("-z", "--zoom", type=float,     help="zooming factor", default=1, metavar='')
    pr.add_argument("-n", "--multirenderer-mode",   help="multi renderer mode: files go to separate renderers", action="store_true")
    pr.add_argument("-s", "--scrolling-mode",       help="scrolling Mode: use slider to scroll files", action="store_true")
    pr.add_argument("-g", "--ray-cast-mode",        help="GPU Ray-casting Mode for 3D image files", action="store_true")
    pr.add_argument("-gx", "--x-spacing", type=float, help="volume x-spacing factor [1]", default=1, metavar='')
    pr.add_argument("-gy", "--y-spacing", type=float, help="volume y-spacing factor [1]", default=1, metavar='')
    pr.add_argument("-gz", "--z-spacing", type=float, help="volume z-spacing factor [1]", default=1, metavar='')
    pr.add_argument("--mode",                       help="volume rendering style (composite/maxproj/...)", default=0, metavar='')
    pr.add_argument("--cmap",                       help="volume rendering color map name", default='jet', metavar='')
    pr.add_argument("-e", "--edit",                 help="free-hand edit the input Mesh", action="store_true")
    pr.add_argument("--slicer2d",                   help="2D Slicer Mode for volumetric data", action="store_true")
    pr.add_argument("--slicer3d",                   help="3D Slicer Mode for volumetric data", action="store_true")
    pr.add_argument("--lego",                       help="voxel rendering for 3D image files", action="store_true")
    pr.add_argument("-r", "--run",                  help="run example from vedo/examples", metavar='')
    pr.add_argument("--search",           type=str, help="search/grep for word in vedo examples", default='', metavar='')
    pr.add_argument("--search-vtk",       type=str, help="search examples for the input vtk class", default='', metavar='')
    pr.add_argument("--reload",                     help="reload the file, ignoring any previous download", action="store_true")
    pr.add_argument("--info", nargs='*',            help="get an info printout of the input file(s)")
    pr.add_argument("--convert", nargs='*',         help="input file(s) to be converted")
    pr.add_argument("--to",               type=str, help="convert to this target format", default='vtk', metavar='')
    pr.add_argument("--image",                      help="image mode for 2d objects", action="store_true")
    pr.add_argument("--eog",                        help="eog-like image visualizer", action="store_true")
    return pr


#################################################################################################
def system_info():
    for i in range(2, len(sys.argv)):
        file = sys.argv[i]
        try:
            A = load(file)
            if isinstance(A, np.ndarray):
                printInfo(A)
            elif isSequence(A):
                for a in A:
                    printInfo(a)
            else:
                printInfo(A)
        except:
            vedo.logger.error(f"Could not load {file}, skip.")

    printc("_" * 65, bold=0)
    printc("vedo version      :", __version__, invert=1, end='   ')
    printc("https://vedo.embl.es", underline=1, italic=1)
    printc("vtk version       :", vtk.vtkVersion().GetVTKVersion())
    printc("python version    :", sys.version.replace("\n", ""))
    printc("python interpreter:", sys.executable)
    printc("vedo installation :", vedo.installdir)
    try:
        import platform
        printc("system            :", platform.system(),
               platform.release(), os.name, platform.machine())
    except:
        pass

    try:
        from screeninfo import get_monitors
        for m in get_monitors():
            pr = '         '
            if m.is_primary:
                pr = '(primary)'
            printc(f"monitor {pr} : {m.name}, resolution=({m.width}, {m.height}), x={m.x}, y={m.y}")
    except:
        printc('monitor           : info is unavailable. Try "pip install screeninfo".')

    try:
        import k3d
        printc("k3d version       :", k3d.__version__, bold=0, dim=1)
    except:
        pass
    try:
        import ipyvtk_simple
        printc("ipyvtk version    :", ipyvtk_simple.__version__, bold=0, dim=1)
    except:
        pass
    try:
        import itkwidgets
        printc("itkwidgets version:", itkwidgets.__version__, bold=0, dim=1)
    except:
        pass
    try:
        import panel
        printc("panel version     :", panel.__version__, bold=0, dim=1)
    except:
        pass


#################################################################################################
def exe_run(args):
    expath = os.path.join(vedo.installdir, "examples", "**", "*.py")
    exfiles = [f for f in glob.glob(expath, recursive=True)]
    f2search = os.path.basename(args.run).lower()
    matching = [s for s in exfiles if (f2search in os.path.basename(s).lower() and "__" not in s)]
    matching = list(sorted(matching))
    nmat = len(matching)
    if nmat == 0:
        printc("No matching example found containing string:", args.run, c=1)
        printc(" Current installation directory is:", vedo.installdir, c=1)
        exit(1)

    if nmat > 1:
        printc("\nSelect one of", nmat, "matching scripts:", c='y', italic=1)
        args.full_screen=True # to print out the one line description

    if args.full_screen: # -f option not to dump the full code but just the first line
        for mat in matching[:25]:
            printc(os.path.basename(mat).replace('.py',''), c='y', italic=1, end=' ')
            with open(mat) as fm:
                lline = ''.join(fm.readlines(60))
                lline = lline.replace('\n',' ').replace('\'','').replace('\"','').replace('-','')
                line = lline[:56] #cut
                if line.startswith('from'): line=''
                if line.startswith('import'): line=''
                if len(lline) > len(line):
                    line += '..'
                if len(line)>5:
                    printc('-', line,  c='y', bold=0, italic=1)
                else:
                    print()

    if nmat>25:
        printc('...', c='y')

    if nmat > 1:
        exit(0)

    if not args.full_screen: # -f option not to dump the full code
        with open(matching[0]) as fm:
            code = fm.read()
        code = "#"*80 + "\n" + code + "\n"+ "#"*80

        try:
            from pygments import highlight
            from pygments.lexers import Python3Lexer
            from pygments.formatters import Terminal256Formatter
            # from pygments.styles import STYLE_MAP
            # print(STYLE_MAP.keys())
            result = highlight(code, Python3Lexer(), Terminal256Formatter(style='zenburn'))
            print(result, end='')

        except:
            printc(code, italic=1, bold=0)
            printc("To colorize code try:  pip install Pygments")
        # print()

    printc("("+matching[0]+")", c='y', bold=0, italic=1)
    os.system('python3 ' + matching[0])

################################################################################################
def exe_convert(args):

    allowedexts = ['vtk', 'vtp', 'vtu', 'vts', 'npy', 'ply', 'stl', 'obj',
                   'byu', 'xml', 'vti', 'tif', 'mhd', 'xml']

    humansort(args.convert)
    nfiles = len(args.convert)
    if nfiles == 0:
        sys.exit()

    target_ext = args.to.lower()

    if target_ext not in allowedexts:
        printc('Sorry target cannot be', target_ext, '\nMust be', allowedexts, c=1)
        sys.exit()

    for f in args.convert:
        source_ext = f.split('.')[-1]

        if target_ext == source_ext:
            continue

        a = load(f)
        newf = f.replace("."+source_ext,"")+"."+target_ext
        a.write(newf, binary=True)

##############################################################################################
def exe_search(args):
    expath = os.path.join(vedo.installdir, "examples", "**", "*.py")
    exfiles = [f for f in sorted(glob.glob(expath, recursive=True))]
    pattern = args.search
    if args.no_camera_share:
        pattern = pattern.lower()
    if len(pattern) > 3:
        for ifile in exfiles:
            with open(ifile, "r") as file:
                fflag=True
                for i,line in enumerate(file):
                    if args.no_camera_share:
                        bline = line.lower()
                    else:
                        bline = line
                    if pattern in bline:
                        if fflag:
                            name = os.path.basename(ifile)
                            etype = ifile.split("/")[-2]
                            printc("--> examples/"+etype+"/"+name+":", c='y', italic=1, invert=1)
                            fflag = False
                        line = line.replace(pattern, "\x1b[4m\x1b[1m"+pattern+"\x1b[0m\u001b[33m")
                        print(f"\u001b[33m{i}\t{line}\x1b[0m", end='')
                        # printc(i, line, c='o', bold=False, end='')
    else:
        printc("Please specify at least four characters.", c='r')

##############################################################################################
def exe_search_vtk(args):
    # input a vtk class name to get links to examples that involve that class
    # From https://kitware.github.io/vtk-examples/site/Python/Utilities/SelectExamples/
    import json
    import tempfile
    from datetime import datetime
    from pathlib import Path
    from urllib.error import HTTPError
    from urllib.request import urlretrieve

    xref_url='https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/Coverage/vtk_vtk-examples_xref.json'

    def download_file(dl_path, dl_url, overwrite=False):
        file_name = dl_url.split('/')[-1]
        # Create necessary sub-directories in the dl_path (if they don't exist).
        Path(dl_path).mkdir(parents=True, exist_ok=True)
        # Download if it doesn't exist in the directory overriding if overwrite is True.
        path = Path(dl_path, file_name)
        if not path.is_file() or overwrite:
            try:
                urlretrieve(dl_url, path)
            except HTTPError as e:
                raise RuntimeError(f'Failed to download {dl_url}. {e.reason}')
        return path

    def get_examples(d, vtk_class, lang, all_values=False, number=5):
        try:
            kv = d[vtk_class][lang].items()
        except KeyError as e:
            print(f'For the combination {vtk_class} and {lang}, this key does not exist: {e}')
            return None, None
        total = len(kv)
        samples = list(kv)
        return total, [f'{s[1]}' for s in samples]

    vtk_class, language, all_values, number = args.search_vtk, "Python", True, 10000
    tmp_dir = tempfile.gettempdir()
    path = download_file(tmp_dir, xref_url, overwrite=False)
    if not path.is_file():
        print(f'The path: {str(path)} does not exist.')

    dt = datetime.today().timestamp() - os.path.getmtime(path)
    # Force a new download if the time difference is > 10 minutes.
    if dt > 600:
        path = download_file(tmp_dir, xref_url, overwrite=True)
    with open(path) as json_file:
        xref_dict = json.load(json_file)

    total_number, examples = get_examples(xref_dict, vtk_class, language, all_values=all_values, number=number)
    if examples:
        if total_number <= number or all_values:
            print(f'VTK Class: {vtk_class}, language: {language}\n'
                  f'Number of example(s): {total_number}.')
        else:
            print(f'VTK Class: {vtk_class}, language: {language}\n'
                  f'Number of example(s): {total_number} with {number} random sample(s) shown.')
        print('\n'.join(examples))
    else:
        print(f'No examples for the VTK Class: {vtk_class} and language: {language}')


#################################################################################################################
def exe_eog(args):
    # print("EOG emulator")
    settings.immediateRendering = False
    settings.useParallelProjection = True
    settings.enableDefaultMouseCallbacks = False
    settings.enableDefaultKeyboardCallbacks = False

    if args.background == "":
        args.background = "white"

    if args.background_grad:
        args.background_grad = getColor(args.background_grad)

    files = []
    for s in sys.argv:
        if '--' in s or s.endswith('.py') or s.endswith('vedo'):
            continue
        if s.endswith('.gif'):
            continue
        files.append(s)


    def vfunc(event):
        # print(event.keyPressed)
        for p in pics:
            if event.keyPressed=="r":
                    p.window(win).level(lev)
            elif event.keyPressed=="Up":
                    p.level(p.level()+10)
            elif event.keyPressed=="Down":
                    p.level(p.level()-10)
            if event.keyPressed=="Right":
                    p.window(p.window()+10)
            elif event.keyPressed=="Down":
                    p.window(p.window()-10)
            elif event.keyPressed=="m":
                    p.mirror()
            elif event.keyPressed=="t":
                    p.rotate(90)
            elif event.keyPressed=="f":
                    p.flip()
            elif event.keyPressed=="b":
                    p.binarize()
            elif event.keyPressed=="i":
                    p.invert()
            elif event.keyPressed=="I":
                    plt.colorPicker(event.picked2d, verbose=True)
            elif event.keyPressed=="k":
                    p.enhance()
            elif event.keyPressed=="s":
                    p.smooth(sigma=1)
            elif event.keyPressed=="S":
                    ahl = plt.hoverLegends[-1]
                    plt.remove(ahl)
                    plt.screenshot() # writer
                    printc("Picture saved as screenshot.png")
                    plt.add(ahl, render=False)
                    return
            elif event.keyPressed=="h":
                printc('---------------------------------------------')
                printc('Press:')
                printc('  up/down     to modify level (or drag mouse)')
                printc('  left/right  to modify window')
                printc('  m           to mirror image horizontally')
                printc('  f           to flip image vertically')
                printc('  t           to rotate image by 90 deg')
                printc('  i           to invert colors')
                printc('  I           to pick the color under mouse')
                printc('  b           to binarize the image')
                printc('  k           to enhance b&w image')
                printc('  s           to apply gaussian smoothing')
                printc('  S           to save image as png')
                printc('---------------------------------------------')

            plt.render()

    pics =[]
    for f in files:
        if os.path.isfile(f):
            try:
                pic = Picture(f)
                if pic:
                    pics.append(pic)
            except:
                vedo.logger.error(f"Could not load image {f}")
        else:
            vedo.logger.error(f"Could not load image {f}")

    n = len(pics)
    if not n:
        return

    pic = pics[0]
    lev, win = pic.level(), pic.window()

    if n > 1:

        plt = Plotter(N=n, sharecam=True, bg=args.background, bg2=args.background_grad)
        plt.addCallback('key press', vfunc)
        for i in range(n):
            p = pics[i].pickable(True)
            pos = [-p.shape[0]/2, -p.shape[1]/2, 0]
            p.pos(pos)
            plt.addHoverLegend(at=i, c='k8', bg='k2', alpha=0.4)
            plt.show(p, axes=0, at=i, mode='image')
        plt.show(interactive=False)
        plt.resetCamera(tight=0.05)
        plt.interactor.Start()

    else:

        shape = pic.shape
        if shape[0]>1500:
            shape[1] = shape[1]/shape[0]*1500
            shape[0]=1500

        if shape[1]>1200:
            shape[0] = shape[0]/shape[1]*1200
            shape[1]=1200

        plt = Plotter(title=files[0], size=shape, bg=args.background, bg2=args.background_grad)
        plt.addCallback('key press', vfunc)
        plt.addHoverLegend(c='k8', bg='k2', alpha=0.4)
        plt.show(pic, mode='image',  interactive=False)
        plt.resetCamera(tight=0.0)
        plt.interactor.Start()

    plt.close()


#################################################################################################################
def draw_scene(args):

    nfiles = len(args.files)
    if nfiles == 0:
        printc("No input files.", c='r')
        return
    humansort(args.files)

    wsize = "auto"
    if args.full_screen:
        wsize = "full"

    if args.ray_cast_mode:
        if args.background == "":
            args.background = "bb"

    if args.background == "":
        args.background = "white"

    if args.background_grad:
        args.background_grad = getColor(args.background_grad)

    if nfiles == 1 and args.files[0].endswith(".gif"): ###can be improved
        frames = load(args.files[0])
        applications.Browser(frames).show(bg=args.background, bg2=args.background_grad)
        return ##########################################################

    if args.scrolling_mode:
        args.multirenderer_mode = False

    N = None
    if args.multirenderer_mode:
        if nfiles < 201:
            N = nfiles
        if nfiles > 200:
            printc("Warning: option '-n' allows a maximum of 200 files", c=1)
            printc("         you are trying to load ", nfiles, " files.\n", c=1)
            N = 200
        plt = Plotter(size=wsize, N=N, bg=args.background, bg2=args.background_grad)
        settings.immediateRendering=False
        plt.axes = args.axes_type
        for i in range(N):
            plt.addHoverLegend(at=i)
        if args.axes_type == 4 or args.axes_type == 5:
            plt.axes = 0
    else:
        N = nfiles
        plt = Plotter(size=wsize, bg=args.background, bg2=args.background_grad)
        plt.axes = args.axes_type
        plt.addHoverLegend()

    plt.sharecam = not args.no_camera_share

    wire = False
    if args.wireframe:
        wire = True

    ##########################################################
    # special case of SLC/TIFF volumes with -g option
    if args.ray_cast_mode:
        # print('DEBUG special case of SLC/TIFF volumes with -g option')

        vol = io.load(args.files[0], force=args.reload)

        if not isinstance(vol, Volume):
            vedo.logger.error(f"expected a Volume but loaded a {type(vol)} object")
            return

        sp = vol.spacing()
        vol.spacing([sp[0]*args.x_spacing, sp[1]*args.y_spacing, sp[2]*args.z_spacing])
        vol.mode(int(args.mode)).color(args.cmap).jittering(True)
        # if args.lighting !='default':
        vol.lighting(args.lighting).jittering(True)
        plt = applications.RayCastPlotter(vol)
        plt.show(viewup="z", interactive=True)
        plt.sliders[0][0].SetEnabled(False)
        plt.sliders[1][0].SetEnabled(False)
        plt.sliders[2][0].SetEnabled(False)
        return

    ##########################################################
    # special case of SLC/TIFF/DICOM volumes with --slicer3d option
    elif args.slicer3d:
        # print('DEBUG special case of SLC/TIFF/DICOM volumes with --slicer3d option')

        useSlider3D = False
        if args.axes_type == 4:
            args.axes_type=1
        elif args.axes_type == 3:
            args.axes_type=1
            useSlider3D = True

        vol = io.load(args.files[0], force=args.reload)

        sp = vol.spacing()
        vol.spacing([sp[0]*args.x_spacing, sp[1]*args.y_spacing, sp[2]*args.z_spacing])

        vedo.plotter_instance = None # reset

        plt = applications.Slicer3DPlotter(
                     vol,
                     bg='white', bg2='lb',
                     useSlider3D=useSlider3D,
                     cmaps=[args.cmap, "Spectral_r", "hot_r", "bone_r", "gist_ncar_r"],
                     alpha=args.alpha,
                     axes=args.axes_type,
                     clamp=True,
                     size=(1000,800),
        )
        return

    ########################################################################
    elif args.edit:
        # print('edit mode for meshes and pointclouds')
        vedo.plotter_instance = None # reset
        settings.useParallelProjection = True

        try:
            m = Mesh(args.files[0], alpha=args.alpha/2, c=args.color)
        except AttributeError:
            vedo.logger.critical("In edit mode, input file must be a point cloud or polygonal mesh.")
            return

        plt = applications.FreeHandCutPlotter(m, splined=True)
        plt.addHoverLegend()
        if not args.background_grad:
            args.background_grad = None
        plt.start(axes=1, bg=args.background, bg2=args.background_grad)

    ########################################################################
    elif args.slicer2d:
        # print('DEBUG special case of SLC/TIFF/DICOM volumes with --slicer2d option')
        vol = io.load(args.files[0], force=args.reload)
        if not vol:
            return
        vol.cmap('bone_r')
        sp = vol.spacing()
        vol.spacing([sp[0]*args.x_spacing, sp[1]*args.y_spacing, sp[2]*args.z_spacing])
        vedo.plotter_instance = None # reset

        plt = applications.Slicer2DPlotter(vol, axes=7)
        plt.close()
        return


    ########################################################################
    # normal mode for single VOXEL file with Isosurface Slider or LEGO mode
    elif nfiles == 1 and (
            ".slc" in args.files[0].lower()
            or ".vti" in args.files[0].lower()
            or ".tif" in args.files[0].lower()
            or ".mhd" in args.files[0].lower()
            or ".nrrd" in args.files[0].lower()
            or ".dem" in args.files[0].lower()
        ):
        # print('DEBUG normal mode for single VOXEL file with Isosurface Slider or LEGO mode')
        vol = io.load(args.files[0], force=args.reload)
        sp = vol.spacing()
        vol.spacing([sp[0]*args.x_spacing, sp[1]*args.y_spacing, sp[2]*args.z_spacing])
        if not args.color:
            args.color = 'gold'
        plt = applications.IsosurfaceBrowser(vol,
                                            lego=args.lego,
                                            c=args.color,
                                            cmap=args.cmap,
                                            delayed=args.lego,
                                            precompute=True,
                                            progress=True,
        )
        plt.show(zoom=args.zoom, viewup="z")
        return


    ########################################################################
    # NORMAL mode for single or multiple files, or multiren mode, or numpy scene
    elif nfiles == 1 or (not args.scrolling_mode):
        # print('DEBUG NORMAL mode for single or multiple files, or multiren mode')

        interactor_mode = 0
        if args.image:
            interactor_mode = 'image'

        ##########################################################
        # loading a full scene
        if ".npy" in args.files[0] or ".npz" in args.files[0] and nfiles == 1:

            objct = io.load(args.files[0], force=args.reload)

            if "Plotter" in str(type(objct)): # loading a full scene
                objct.show(mode=interactor_mode)
                return
            else:                             # loading a set of meshes
                plt.show(objct, mode=interactor_mode)
                return
        #########################################################

        ds=0
        actors = []
        for i in range(N):
            f = args.files[i]

            colb = args.color
            if args.color is None and N > 1:
                colb = i

            actor = load(f, force=args.reload)

            if isinstance(actor, (TetMesh, UGrid)):
                actor = actor.tomesh().shrink(0.975).c(colb).alpha(args.alpha)

            if isinstance(actor, Mesh):
                actors.append(actor)
                actor.c(colb).alpha(args.alpha).wireframe(wire).lighting(args.lighting)
                if args.flat:
                    actor.flat()
                else:
                    actor.phong()

                if i==0 and args.texture_file:
                    actor.texture(args.texture_file)

                if args.point_size > 0:
                    try:
                        actor.GetProperty().SetPointSize(args.point_size)
                        actor.GetProperty().SetRepresentationToPoints()
                    except AttributeError:
                        pass

                if args.showedges:
                    try:
                        actor.GetProperty().SetEdgeVisibility(1)
                        actor.GetProperty().SetLineWidth(0.1)
                        actor.GetProperty().SetRepresentationToSurface()
                    except AttributeError:
                        pass
            else:
                actors.append(actor)

            if args.multirenderer_mode:
                try:
                    ds = actor.diagonalSize() * 3
                    plt.camera.SetClippingRange(0, ds)
                    plt.show(actor, at=i, interactive=False, zoom=args.zoom, mode=interactor_mode)
                    plt.actors = actors
                    # if args.no_camera_share: ## BUG
                        # plt.resetCamera()
                        # plt.renderers[i].ResetCameraClippingRange()
                        # print([plt.camera])
                except AttributeError:
                    # wildcards in quotes make glob return actor as a list :(
                    vedo.logger.error("Please do not use wildcards within single or double quotes")

        if args.multirenderer_mode:
            plt.interactor.Start()

        else:

            # scene is empty
            if all(a is None for a in actors):
                vedo.logger.error("Could not load file(s). Quit.")
                return

            plt.show(actors, interactive=True, zoom=args.zoom, mode=interactor_mode)
        return

    ########################################################################
    # scrolling mode  -s
    else:
        #print("DEBUG simple browser mode  -s")
        if plt.axes==4:
            plt.axes=1

        acts = plt.load(args.files, force=args.reload)
        for a in acts:
            if hasattr(a, 'c'): #Picture doesnt have it
                a.c(args.color)
            a.alpha(args.alpha)

        plt = applications.Browser(acts, axes=1)
        plt.show(zoom=args.zoom).close()


########################################################################
def exe_gui(args):

    # print('DEBUG gui started')
    if sys.version_info[0] > 2:
        from tkinter import Frame, Tk, BOTH, Label, Scale, Checkbutton, BooleanVar, StringVar
        from tkinter.ttk import Button, Style, Combobox, Entry
        from tkinter import filedialog as tkFileDialog
    else:
        from Tkinter import Frame, Tk, BOTH, Label, Scale, Checkbutton, BooleanVar, StringVar
        from ttk import Button, Style, Combobox, Entry
        import tkFileDialog

    ######################
    class vedoGUI(Frame):
        def __init__(self, parent):
            Frame.__init__(self, parent, bg="white")
            self.parent = parent
            self.filenames = []
            self.noshare = BooleanVar()
            self.flat = BooleanVar()
            self.xspacing = StringVar()
            self.yspacing = StringVar()
            self.zspacing = StringVar()
            self.background_grad = BooleanVar()
            self.initUI()

        def initUI(self):
            self.parent.title("vedo")
            self.style = Style()
            self.style.theme_use("clam")
            self.pack(fill=BOTH, expand=True)

            ############import
            Button(self, text="Import Files", command=self._importCMD, width=15).place(x=115, y=17)

            ############meshes
            Frame(root, height=1, width=398, bg="grey").place(x=1, y=60)
            Label(self, text="Meshes", fg="white", bg="green", font=("Courier 11 bold")).place(x=20, y=65)

            # color
            Label(self, text="Color:", bg="white").place(x=30, y=98)
            colvalues = ('by scalar', 'gold','red','green','blue', 'coral','plum','tomato')
            self.colorCB = Combobox(self, state="readonly", values=colvalues, width=10)
            self.colorCB.current(0)
            self.colorCB.place(x=100, y=98)

            # mode
            modvalues = ('surface', 'surf. & edges','wireframe','point cloud')
            self.surfmodeCB = Combobox(self, state="readonly", values=modvalues, width=14)
            self.surfmodeCB.current(0)
            self.surfmodeCB.place(x=205, y=98)

            # alpha
            Label(self, text="Alpha:", bg="white").place(x=30, y=145)
            self.alphaCB = Scale(
                self,
                from_=0,
                to=1,
                resolution=0.02,
                bg="white",
                length=220,
                orient="horizontal",
            )
            self.alphaCB.set(1.0)
            self.alphaCB.place(x=100, y=125)

            # lighting
            Label(self, text="Lighting:", bg="white").place(x=30, y=180)
            lightvalues = ('default','metallic','plastic','shiny','glossy')
            self.lightCB = Combobox(self, state="readonly", values=lightvalues, width=10)
            self.lightCB.current(0)
            self.lightCB.place(x=100, y=180)
            # shading phong or flat
            self.flatCB = Checkbutton(self, text="flat shading", var=self.flat, bg="white")
            #self.flatCB.select()
            self.flatCB.place(x=210, y=180)

            # rendering arrangement
            Label(self, text="Arrange as:", bg="white").place(x=30, y=220)
            schemevalues = ('superpose (default)','mesh browser', 'n sync-ed renderers')
            self.schememodeCB = Combobox(self, state="readonly", values=schemevalues, width=20)
            self.schememodeCB.current(0)
            self.schememodeCB.place(x=160, y=220)

            # share cam
            self.noshareCB = Checkbutton(self, text="independent cameras",
                                         variable=self.noshare, bg="white")
            self.noshareCB.place(x=160, y=245)


            ############volumes
            Frame(root, height=1, width=398, bg="grey").place(x=1, y=275)
            Label(self, text="Volumes", fg="white", bg="blue", font=("Courier 11 bold")).place(x=20, y=280)

            # mode
            Label(self, text="Rendering mode:", bg="white").place(x=30, y=310)
            modevalues = (
                "isosurface (default)",
                "composite",
                "maximum proj",
                "lego",
                "slicer2d",
                "slicer3d",
            )
            self.modeCB = Combobox(self, state="readonly", values=modevalues, width=20)
            self.modeCB.current(0)
            self.modeCB.place(x=160, y=310)

            Label(self, text="Spacing factors:", bg="white").place(x=30, y=335)
            self.xspacingCB = Entry(self, textvariable=self.xspacing, width=3)
            self.xspacing.set('1.0')
            self.xspacingCB.place(x=160, y=335)
            self.yspacingCB = Entry(self, textvariable=self.yspacing, width=3)
            self.yspacing.set('1.0')
            self.yspacingCB.place(x=210, y=335)
            self.zspacingCB = Entry(self, textvariable=self.zspacing, width=3)
            self.zspacing.set('1.0')
            self.zspacingCB.place(x=260, y=335)


            ############## options
            Frame(root, height=1, width=398,bg="grey").place(x=1, y=370)
            Label(self, text="Options", fg='white', bg="brown", font=("Courier 11 bold")).place(x=20, y=375)

            # backgr color
            Label(self, text="Background color:", bg="white").place(x=30, y=405)
            bgcolvalues = ("white", "lightyellow", "azure", "blackboard", "black")
            self.bgcolorCB = Combobox(self, state="readonly", values=bgcolvalues, width=9)
            self.bgcolorCB.current(3)
            self.bgcolorCB.place(x=160, y=405)
            # backgr color gradient
            self.backgroundGradCB = Checkbutton(self, text="gradient",
                                                variable=self.background_grad, bg="white")
            self.backgroundGradCB.place(x=255, y=405)

            ################ render button
            Frame(root, height=1, width=398, bg="grey").place(x=1, y=437)
            Button(self, text="Render", command=self._run, width=15).place(x=115, y=454)


        def _importCMD(self):
            ftypes = [
                ("All files", "*"),
                ("VTK files", "*.vtk"),
                ("VTK files", "*.vtp"),
                ("VTK files", "*.vts"),
                ("VTK files", "*.vtu"),
                ("Surface Mesh", "*.ply"),
                ("Surface Mesh", "*.obj"),
                ("Surface Mesh", "*.stl"),
                ("Surface Mesh", "*.off"),
                ("Surface Mesh", "*.facet"),
                ("Volume files", "*.tif"),
                ("Volume files", "*.slc"),
                ("Volume files", "*.vti"),
                ("Volume files", "*.mhd"),
                ("Volume files", "*.nrrd"),
                ("Volume files", "*.nii"),
                ("Volume files", "*.dem"),
                ("Picture files", "*.png"),
                ("Picture files", "*.jpg"),
                ("Picture files", "*.bmp"),
                ("Picture files", "*.gif"),
                ("Picture files", "*.jpeg"),
                ("Geojson files", "*.geojson"),
                ("DOLFIN files", "*.xml.gz"),
                ("DOLFIN files", "*.xml"),
                ("DOLFIN files", "*.xdmf"),
                ("Neutral mesh", "*.neu*"),
                ("GMESH", "*.gmsh"),
                ("Point Cloud", "*.pcd"),
                ("3DS", "*.3ds"),
                ("Numpy scene file", "*.npy"),
                ("Numpy scene file", "*.npz"),
            ]
            self.filenames = tkFileDialog.askopenfilenames(parent=root, filetypes=ftypes)
            args.files = list(self.filenames)


        def _run(self):

            args.files = list(self.filenames)
            if self.colorCB.get() == "by scalar":
                args.color = None
            else:
                if self.colorCB.get() == 'red':
                    args.color = 'crimson'
                elif self.colorCB.get() == 'green':
                    args.color = 'limegreen'
                elif self.colorCB.get() == 'blue':
                    args.color = 'darkcyan'
                else:
                    args.color = self.colorCB.get()

            args.alpha = self.alphaCB.get()

            args.wireframe = False
            args.showedges = False
            args.point_size = 0
            if self.surfmodeCB.get() == 'point cloud':
                args.point_size = 2
            elif self.surfmodeCB.get() == 'wireframe':
                args.wireframe = True
            elif self.surfmodeCB.get() == 'surf. & edges':
                args.showedges = True
            else:
                pass # normal surface mode

            args.lighting = self.lightCB.get()
            args.flat = self.flat.get()

            args.no_camera_share = self.noshare.get()
            args.background = self.bgcolorCB.get()

            args.background_grad = None
            if self.background_grad.get():
                b = getColor(args.background)
                args.background_grad = (b[0]/1.8, b[1]/1.8, b[2]/1.8)

            args.multirenderer_mode = False
            args.scrolling_mode = False
            if self.schememodeCB.get() == "n sync-ed renderers":
                args.multirenderer_mode = True
            elif self.schememodeCB.get() == "mesh browser":
                args.scrolling_mode = True

            args.ray_cast_mode = False
            args.lego = False
            args.slicer3d = False
            args.slicer2d = False
            args.lego = False
            args.mode = 0
            if self.modeCB.get() == "composite":
                args.ray_cast_mode = True
                args.mode = 0
            elif self.modeCB.get() == "maximum proj":
                args.ray_cast_mode = True
                args.mode = 1
            elif self.modeCB.get() == "slicer3d":
                args.slicer3d = True
            elif self.modeCB.get() == "slicer2d":
                args.slicer2d = True
            elif self.modeCB.get() == "lego":
                args.lego = True

            args.x_spacing = 1
            args.y_spacing = 1
            args.z_spacing = 1
            if self.xspacing.get() != '1.0': args.x_spacing = float(self.xspacing.get())
            if self.yspacing.get() != '1.0': args.y_spacing = float(self.yspacing.get())
            if self.zspacing.get() != '1.0': args.z_spacing = float(self.zspacing.get())

            draw_scene(args)
            if os.name == "nt":
                exit()
            if vedo.plotter_instance:
                vedo.plotter_instance.close()

    root = Tk()
    root.geometry("360x500")
    app = vedoGUI(root)

    def tkcallback(event):
        #printc("xy cursor position:", event.x, event.y, event.char)
        if event.char == 'q':
            root.destroy()

    app.bind("<Key>", tkcallback)
    app.focus_set()
    app.pack()

    if os.name == "nt" and len(sys.argv) > 1:
        app.filenames = sys.argv[1:]
        print("Already", len(app.filenames), "files loaded.")

    root.mainloop()
