#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Command Line Interface module
-----------------------------
The library includes a handy Command Line Interface.

    # For a list of options type:

    vedo -h

    # Some useful bash aliases:
    #
    alias v='vedo '
    alias vr='vedo --run '             # to search and run examples by name
    alias vs='vedo -i --search '       # to search for a string in examples
    alias vdoc='vedo -i --search-code' # to search for a string in source code
    alias ve='vedo --eog '             # to view single and multiple images
    #
    alias vv='vedo -bg blackboard -bg2 gray3 -z 1.05 -k glossy -c blue9 '
"""
import argparse
import glob
import os
import sys
import numpy as np

from vedo.utils import humansort
from vedo.colors import get_color, printc
from vedo.mesh import Mesh
from vedo.image import Image
from vedo.plotter import Plotter
from vedo.grids import UnstructuredGrid, TetMesh
from vedo.volume import Volume

import vedo
from vedo import __version__
from vedo import file_io, load, settings
from vedo import applications

__all__ = []

##############################################################################################
def execute_cli():

    parser = get_parser()
    args = parser.parse_args()

    if "/vedo/vedo" in vedo.installdir:
        vedo.installdir = vedo.installdir.replace("vedo/", "")

    if "\\vedo\\vedo" in vedo.installdir:
        vedo.installdir = vedo.installdir.replace("vedo\\", "")

    if args.info is not None:
        system_info()

    elif args.run:
        exe_run(args)

    elif args.search:
        exe_search(args)

    elif args.search_vtk:
        exe_search_vtk(args)

    elif args.search_code:
        exe_search_code(args)

    elif args.convert:
        exe_convert(args)

    elif args.eog:
        exe_eog(args)

    elif len(args.files) == 0:
        system_info()
        printc(
            ":idea: No input files? Try:\n vedo https://vedo.embl.es/examples/data/panther.stl.gz",
            c="y",
        )

    else:
        draw_scene(args)


##############################################################################################
def get_parser():

    descr = f"version {__version__}"
    descr += " - check out home page at https://vedo.embl.es"

    pr = argparse.ArgumentParser(description=descr)
    pr.add_argument('files', nargs='*',             help="input filename(s)")
    pr.add_argument("-c", "--color", type=str,      help="mesh color [integer or color name]", default=None, metavar='')
    pr.add_argument("-a", "--alpha",    type=float, help="alpha value [0-1]", default=1, metavar='')
    pr.add_argument("-w", "--wireframe",            help="use wireframe representation", action="store_true")
    pr.add_argument("-p", "--point-size", type=float, help="specify point size", default=-1, metavar='')
    pr.add_argument("-l", "--showedges",            help="show a thin line on mesh edges", action="store_true")
    pr.add_argument("-k", "--lighting", type=str,   help="metallic, plastic, shiny, glossy or off", default='default', metavar='')
    pr.add_argument("-K", "--flat",                 help="use flat shading", action="store_true")
    pr.add_argument("-t", "--texture-file",         help="texture image file", default='', metavar='')
    pr.add_argument("-x", "--axes-type", type=int,  help="specify axes type [0-14]", default=1, metavar='')
    pr.add_argument("-i", "--no-camera-share",      help="do not share camera in renderers", action="store_true")
    pr.add_argument("-f", "--full-screen",          help="full screen mode", action="store_true")
    pr.add_argument("-bg","--background", type=str, help="background color [integer or color name]", default='', metavar='')
    pr.add_argument("-bg2", "--background-grad",    help="use background color gradient", default='', metavar='')
    pr.add_argument("-z", "--zoom", type=float,     help="zooming factor", default=1, metavar='')
    pr.add_argument("-n", "--multirenderer-mode",   help="multi renderer mode: files go to separate renderers", action="store_true")
    pr.add_argument("-s", "--sequence-mode",        help="sequence mode: use slider to browse files", action="store_true")
    pr.add_argument("-g", "--ray-cast-mode",        help="GPU Ray-casting Mode for 3D image files", action="store_true")
    pr.add_argument("-gx", "--x-spacing", type=float, help="volume x-spacing factor [1]", default=1, metavar='')
    pr.add_argument("-gy", "--y-spacing", type=float, help="volume y-spacing factor [1]", default=1, metavar='')
    pr.add_argument("-gz", "--z-spacing", type=float, help="volume z-spacing factor [1]", default=1, metavar='')
    pr.add_argument("--mode",                       help="volume rendering style (composite/maxproj/...)", default=0, metavar='')
    pr.add_argument("--cmap",                       help="volume rendering color map name", default='jet', metavar='')
    pr.add_argument("-e", "--edit",                 help="free-hand edit the input Mesh", action="store_true")
    pr.add_argument("--slicer2d",                   help="2D Slicer Mode for volumetric data", action="store_true")
    pr.add_argument("--slicer3d",                   help="3D Slicer Mode for volumetric data", action="store_true")
    pr.add_argument("-r", "--run",                  help="run example from vedo/examples", metavar='')
    pr.add_argument("--search",           type=str, help="search/grep for word in vedo examples", default='', metavar='')
    pr.add_argument("--search-vtk",       type=str, help="search examples for the input vtk class", default='', metavar='')
    pr.add_argument("--search-code",      type=str, help="search keyword in source code", default='', metavar='')
    pr.add_argument("--reload",                     help="reload the file, ignoring any previous download", action="store_true")
    pr.add_argument("--info", nargs='*',            help="get an info printout of the current installation")
    pr.add_argument("--convert", nargs='*',         help="input file(s) to be converted")
    pr.add_argument("--to",               type=str, help="convert to this target format", default='vtk', metavar='')
    pr.add_argument("--image",                      help="image mode for 2d objects", action="store_true")
    pr.add_argument("--eog",                        help="eog-like image visualizer", action="store_true")
    pr.add_argument("--font",                       help="font name", default="Normografo", metavar='')
    return pr


#################################################################################################
def system_info():
    from vtkmodules.all import vtkVersion

    printc(f"vedo version      : {__version__}  (https://vedo.embl.es) ".ljust(65), invert=1)
    printc("vtk version       :", vtkVersion().GetVTKVersion())
    printc("numpy version     :", np.__version__)
    printc("python version    :", sys.version.replace("\n", ""))
    printc("python interpreter:", sys.executable)
    printc("installation point:", vedo.installdir[:70])
    try:
        import platform
        printc(
            "system            :",
            platform.system(),
            platform.release(),
            os.name,
            platform.machine(),
        )
    except ModuleNotFoundError:
        pass

    try:
        import k3d
        printc("k3d version       :", k3d.__version__, bold=0, dim=1)
    except ModuleNotFoundError:
        pass

    # try:
    #     import trame
    #     printc("trame version     :", trame.__version__, bold=0, dim=1)
    # except ModuleNotFoundError:
    #     pass


#################################################################################################
def exe_run(args):
    expath = os.path.join(vedo.installdir, "examples", "**", "*.py")
    exfiles = list(glob.glob(expath, recursive=True))
    f2search = os.path.basename(args.run).lower()
    matching = [
        s
        for s in exfiles
        if (
            f2search in os.path.basename(s).lower()
            and "__" not in s
            and "dolfin" not in s
            and "trimesh" not in s
        )
    ]
    matching = list(sorted(matching))
    nmat = len(matching)
    if nmat == 0:
        printc(f":sad: No matching example with name: {args.run}", c="y")
        # Nothing found, try to search for a script content:
        args.search = args.run
        args.run = ""
        exe_search(args)
        return

    if nmat > 1:
        printc(f":target: Found {nmat} scripts containing string '{args.run}':", c="c")
        args.full_screen = True  # to print out the one line description

    if args.full_screen:  # -f option not to dump the full code but just the first line
        for mat in matching[:30]:
            printc(os.path.basename(mat).replace(".py", ""), c="c", end=" ")
            with open(mat, "r", encoding="UTF-8") as fm:
                lline = "".join(fm.readlines(60))
                maxidx1 = lline.find("import ")
                maxidx2 = lline.find("from vedo")
                maxid = min(maxidx1, maxidx2)
                lline = lline[:maxid]  # cut where the code starts
                lline = lline.replace("\n", " ").replace("'", "").replace('"', "")
                lline = lline.replace("#", "").replace("-", "").replace("  ", " ")
                line = lline[:68]  # cut long lines
                if len(lline) > len(line) + 1:
                    line += ".."
                if len(line) > 5:
                    printc("-", line, c="c", bold=0, italic=1, dim=1)
                else:
                    print()

    if nmat > 30:
        printc(f"... (and {nmat-30} more)", c="c")

    if nmat > 1:
        printc(":idea: Type 'vedo -r <name>' to run one of them", bold=0, c="c")
        return

    if not args.full_screen:  # -f option not to dump the full code
        with open(matching[0], "r", encoding="UTF-8") as fm:
            code = fm.read()
        code = "#" * 80 + "\n" + code + "\n" + "#" * 80

        from pygments import highlight
        from pygments.lexers import Python3Lexer
        from pygments.formatters import Terminal256Formatter

        # from pygments.styles import STYLE_MAP
        # print(STYLE_MAP.keys())
        result = highlight(code, Python3Lexer(), Terminal256Formatter(style="zenburn"))
        print(result, end="")

    printc("(" + matching[0] + ")", c="y", bold=0, italic=1)
    os.system(f"python {matching[0]} || python3 {matching[0]}")


################################################################################################
def exe_convert(args):

    allowed_exts = [
        "vtk",
        "vtp",
        "vtu",
        "vts",
        "npy",
        "ply",
        "stl",
        "obj",
        "byu",
        "xml",
        "vti",
        "tif",
        "mhd",
        "xml",
    ]

    humansort(args.convert)
    nfiles = len(args.convert)
    if nfiles == 0:
        sys.exit()

    target_ext = args.to.lower()

    if target_ext not in allowed_exts:
        printc(f":sad: Sorry target cannot be {target_ext}\nMust be {allowed_exts}", c='r')
        sys.exit()

    for f in args.convert:
        source_ext = f.split(".")[-1]

        if target_ext == source_ext:
            continue

        a = load(f)
        newf = f.replace("." + source_ext, "") + "." + target_ext
        a.write(newf, binary=True)


##############################################################################################
def exe_search(args):
    expath = os.path.join(vedo.installdir, "examples", "**", "*.py")
    exfiles = list(sorted(glob.glob(expath, recursive=True)))
    pattern = args.search
    if args.no_camera_share:
        pattern = pattern.lower()
    if len(pattern) > 3:
        for ifile in exfiles:
            if "dolfin" in ifile:
                continue
            if "trimesh" in ifile:
                continue
            with open(ifile, "r", encoding="UTF-8") as file:
                fflag = True
                for i, line in enumerate(file):
                    if args.no_camera_share:
                        bline = line.lower()
                    else:
                        bline = line
                    if pattern in bline:
                        if fflag:
                            name = os.path.basename(ifile)
                            try:
                                etype = ifile.split("/")[-2]
                                printc(
                                    "--> examples/" + etype + "/" + name + ":",
                                    c="y",
                                    italic=1,
                                    invert=1,
                                )
                            except IndexError:
                                etype = ifile.split("\\")[-2]
                                printc(
                                    "--> examples\\" + etype + "\\" + name + ":",
                                    c="y",
                                    italic=1,
                                    invert=1,
                                )
                            fflag = False
                        line = line.replace(
                            pattern, "\x1b[4m\x1b[1m" + pattern + "\x1b[0m\u001b[33m"
                        )
                        print(f"\u001b[33m{i}\t{line}\x1b[0m", end="")
                        # printc(i, line, c='y', bold=False, end='')
    else:
        printc("Please use at least 4 letters in keyword search!", c="r")


##############################################################################################
def exe_search_code(args):

    import inspect
    from pygments import highlight
    from pygments.lexers import Python3Lexer
    from pygments.formatters import Terminal256Formatter

    # styles: autumn, material, rrt, zenburn
    style = "zenburn"
    key = args.search_code
    iopt = args.no_camera_share
    if key.lower() == key:
        iopt = True

    if len(key) < 4:
        printc("Please use at least 4 letters in keyword search!", c="r")
        return

    def _dump(mcontent):
        for name, mm in mcontent:
            if name.startswith("_"):
                continue
            if name.startswith("vtk"):
                continue
            # if not inspect.isfunction(mm):
            #     continue

            try:
                mmdoc = inspect.getsource(mm)
            except TypeError:
                return

            if mmdoc is None:
                continue

            if iopt:
                # -i option to ignore case
                mmdoc_lower = mmdoc.lower()
                key_lower = key.lower()
                name_lower = name.lower()
            else:
                mmdoc_lower = mmdoc
                key_lower = key
                name_lower = name

            if "eprecated" in mmdoc_lower:
                continue

            if key_lower in name_lower:

                sname = inspect.getmodule(mm).__name__ + " -> " + name
                if sname in snames:
                    continue
                snames.append(sname)

                printc(
                    ":checked:Found matching",
                    mm,
                    "in module",
                    os.path.basename(inspect.getfile(mm)),
                    c="y",
                    invert=True,
                )
                mmdoc = mmdoc.replace("``", '"').replace("`", '"')
                mmdoc = mmdoc.replace(".. warning::", "Warning!")
                result = highlight(mmdoc, Python3Lexer(), Terminal256Formatter(style=style))
                idcomment = result.rfind('"""')
                print(result[: idcomment + 3], "\x1b[0m\n")

    # printc("..parsing source code, please wait", c="y", bold=False)
    content = inspect.getmembers(vedo)
    snames = []
    for name, m in content:
        if name.startswith("_"):
            continue
        if not inspect.isclass(m) and not inspect.isfunction(m):
            continue
        if inspect.isbuiltin(m):
            continue

        # if name != "Points": continue # test
        # printc("---", name, str(m), c='r')

        # function case
        _dump([[name, m]])

        # class case
        mcontent = inspect.getmembers(m)
        _dump(mcontent)


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

    xref_url = "https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/Coverage/vtk_vtk-examples_xref.json"

    def _download_file(dl_path, dl_url, overwrite=False):
        file_name = dl_url.split("/")[-1]
        # Create necessary sub-directories in the dl_path (if they don't exist).
        Path(dl_path).mkdir(parents=True, exist_ok=True)
        # Download if it doesn't exist in the directory overriding if overwrite is True.
        path = Path(dl_path, file_name)
        if not path.is_file() or overwrite:
            try:
                urlretrieve(dl_url, path)
            except HTTPError as e:
                raise RuntimeError(f"Failed to download {dl_url}. {e.reason}")
        return path

    def _get_examples(d, vtk_class, lang):
        try:
            kv = d[vtk_class][lang].items()
        except KeyError as e:
            print(f"For the combination {vtk_class} and {lang}, this key does not exist: {e}")
            return None, None
        total = len(kv)
        samples = list(kv)
        return total, [f"{s[1]}" for s in samples]

    vtk_class, language, all_values, number = args.search_vtk, "Python", True, 10000
    tmp_dir = tempfile.gettempdir()
    path = _download_file(tmp_dir, xref_url, overwrite=False)
    if not path.is_file():
        print(f"The path: {str(path)} does not exist.")

    dt = datetime.today().timestamp() - os.path.getmtime(path)
    # Force a new download if the time difference is > 10 minutes.
    if dt > 600:
        path = _download_file(tmp_dir, xref_url, overwrite=True)
    with open(path, "r", encoding="UTF-8") as json_file:
        xref_dict = json.load(json_file)

    total_number, examples = _get_examples(xref_dict, vtk_class, language)
    if examples:
        if total_number <= number or all_values:
            print(
                f"VTK Class: {vtk_class}, language: {language}\n"
                f"Number of example(s): {total_number}."
            )
        else:
            print(
                f"VTK Class: {vtk_class}, language: {language}\n"
                f"Number of example(s): {total_number} with {number} random sample(s) shown."
            )
        print("\n".join(examples))
    else:
        print(f"No examples for the VTK Class: {vtk_class} and language: {language}")


#################################################################################################################
def exe_eog(args):
    # print("EOG emulator")
    if settings.dry_run_mode >= 2:
        print(f"EOG emulator in dry run mode {settings.dry_run_mode}. Skip.")
        return
    settings.immediate_rendering = False
    settings.use_parallel_projection = True
    settings.enable_default_mouse_callbacks = False
    settings.enable_default_keyboard_callbacks = False

    if args.background == "":
        args.background = "white"

    if args.background_grad:
        args.background_grad = get_color(args.background_grad)

    files = []
    for s in sys.argv:
        if "--" in s or s.endswith(".py") or s.endswith("vedo"):
            continue
        if s.endswith(".gif"):
            continue
        files.append(s)

    def vfunc(event):
        # print(event.keypress)
        for p in pics:
            if event.keypress == "r":
                p.window(win).level(lev)
            elif event.keypress == "Up":
                p.level(p.level() + 10)
            elif event.keypress == "Down":
                p.level(p.level() - 10)
            if event.keypress == "Right":
                p.window(p.window() + 10)
            elif event.keypress == "Down":
                p.window(p.window() - 10)
            elif event.keypress == "m":
                p.mirror()
            elif event.keypress == "t":
                p.rotate(90)
            elif event.keypress == "f":
                p.flip()
            elif event.keypress == "b":
                p.binarize()
            elif event.keypress == "i":
                p.invert()
            elif event.keypress == "I":
                plt.color_picker(event.picked2d, verbose=True)
            elif event.keypress == "k":
                p.enhance()
            elif event.keypress == "s":
                p.smooth(sigma=1)
            elif event.keypress == "S":
                ahl = plt.hover_legends[-1]
                plt.remove(ahl)
                plt.screenshot()  # writer
                printc(":camera: Image saved as screenshot.png")
                plt.add(ahl)
                return
            elif event.keypress == "h":
                printc("---------------------------------------------")
                printc("Press:")
                printc("  up/down     to modify level (or drag mouse)")
                printc("  left/right  to modify window")
                printc("  m           to mirror image horizontally")
                printc("  f           to flip image vertically")
                printc("  t           to rotate image by 90 deg")
                printc("  i           to invert colors")
                printc("  I           to pick the color under mouse")
                printc("  b           to binarize the image")
                printc("  k           to enhance b&w image")
                printc("  s           to apply gaussian smoothing")
                printc("  S           to save image as png")
                printc("---------------------------------------------")

            plt.render()

    pics = []
    for f in files:
        if os.path.isfile(f):
            try:
                pic = Image(f)
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
        plt.add_callback("key press", vfunc)
        for i in range(n):
            p = pics[i].pickable(True)
            pos = [-p.shape[0] / 2, -p.shape[1] / 2, 0]
            p.pos(pos)
            plt.add_hover_legend(at=i, c="k8", bg="k2", alpha=0.4)
            plt.show(p, axes=0, at=i, mode="image")
        plt.show(interactive=False)
        plt.reset_camera(tight=0.05)
        plt.interactor.Start()
        if vedo.vtk_version == (9, 2, 2):
            plt.interactor.GetRenderWindow().SetDisplayId("_0_p_void")

    else:

        shape = pic.shape
        if shape[0] > 1500:
            shape[1] = shape[1] / shape[0] * 1500
            shape[0] = 1500

        if shape[1] > 1200:
            shape[0] = shape[0] / shape[1] * 1200
            shape[1] = 1200

        plt = Plotter(title=files[0], size=shape, bg=args.background, bg2=args.background_grad)
        plt.add_callback("key press", vfunc)
        plt.add_hover_legend(c="k8", bg="k2", alpha=0.4)
        plt.show(pic, mode="image", interactive=False)
        plt.reset_camera(tight=0.0)
        plt.interactor.Start()
        if vedo.vtk_version == (9, 2, 2):
            plt.interactor.GetRenderWindow().SetDisplayId("_0_p_void")

    plt.close()


#################################################################################################################
def draw_scene(args):
    if settings.dry_run_mode >= 2:
        print(f"draw_scene called in dry run mode {settings.dry_run_mode}. Skip.")
        return

    nfiles = len(args.files)
    if nfiles == 0:
        printc("No input files.", c="r")
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
        args.background_grad = get_color(args.background_grad)

    if nfiles == 1 and args.files[0].endswith(".gif"):  ###can be improved
        frames = load(args.files[0])
        applications.Browser(frames).show(bg=args.background, bg2=args.background_grad)
        return  ##########################################################

    if args.sequence_mode:
        args.multirenderer_mode = False
    settings.default_font = args.font

    sharecam = args.no_camera_share

    N = None
    if args.multirenderer_mode:
        if nfiles < 201:
            N = nfiles
        if nfiles > 200:
            printc(":lightning: Warning: option '-n' allows a maximum of 200 files", c="y")
            printc("         you are trying to load ", nfiles, " files.\n", c="y")
            N = 200
        if N > 4:
            settings.use_depth_peeling = False

        plt = Plotter(
            size=wsize,
            N=N,
            bg=args.background,
            bg2=args.background_grad,
            sharecam=sharecam,
        )
        settings.immediate_rendering = False
        plt.axes = args.axes_type
        for i in range(N):
            plt.add_hover_legend(at=i)
        if args.axes_type in (4, 5):
            plt.axes = 0
    else:
        N = nfiles
        plt = Plotter(size=wsize, bg=args.background, bg2=args.background_grad)
        plt.axes = args.axes_type
        plt.add_hover_legend()

    ##########################################################
    # special case of SLC/TIFF volumes with -g option
    if args.ray_cast_mode:
        # print('DEBUG special case of SLC/TIFF volumes with -g option')

        vol = file_io.load(args.files[0], force=args.reload)

        if not isinstance(vol, Volume):
            vedo.logger.error(f"expected a Volume but loaded a {type(vol)} object")
            return

        sp = vol.spacing()
        vol.spacing([sp[0] * args.x_spacing, sp[1] * args.y_spacing, sp[2] * args.z_spacing])
        vol.mode(int(args.mode)).color(args.cmap).jittering(True)
        plt = applications.RayCastPlotter(vol)
        plt.show(viewup="z", interactive=True).close()
        return

    ##########################################################
    # special case of SLC/TIFF/DICOM volumes with --slicer3d option
    elif args.slicer3d:
        # print('DEBUG special case of SLC/TIFF/DICOM volumes with --slicer3d option')

        useSlider3D = False
        if args.axes_type == 4:
            args.axes_type = 1
        elif args.axes_type == 3:
            args.axes_type = 1
            useSlider3D = True

        vol = file_io.load(args.files[0], force=args.reload)

        sp = vol.spacing()
        vol.spacing([sp[0] * args.x_spacing, sp[1] * args.y_spacing, sp[2] * args.z_spacing])

        vedo.plotter_instance = None  # reset

        plt = applications.Slicer3DPlotter(
            vol,
            bg="white",
            bg2="lb",
            use_slider3d=useSlider3D,
            axes=args.axes_type,
            clamp=True,
            size=(1350, 1000),
        )
        plt += vedo.Text2D(args.files[0], pos="top-left", font="VictorMono", s=1, c="k")
        plt.show()
        return

    ########################################################################
    elif args.edit:
        # print('edit mode for meshes and pointclouds')
        vedo.plotter_instance = None  # reset
        settings.use_parallel_projection = True

        try:
            m = Mesh(args.files[0], alpha=args.alpha / 2, c=args.color)
        except AttributeError:
            vedo.logger.critical(
                "In edit mode, input file must be a point cloud or polygonal mesh."
            )
            return

        plt = applications.FreeHandCutPlotter(m, splined=True)
        plt.add_hover_legend()
        if not args.background_grad:
            args.background_grad = None
        plt.start(axes=1, bg=args.background, bg2=args.background_grad)

    ########################################################################
    elif args.slicer2d:
        # print('DEBUG special case of SLC/TIFF/DICOM volumes with --slicer2d option')
        vol = file_io.load(args.files[0], force=args.reload)
        if not vol:
            return
        vol.cmap("bone_r")
        sp = vol.spacing()
        vol.spacing([sp[0] * args.x_spacing, sp[1] * args.y_spacing, sp[2] * args.z_spacing])
        vedo.plotter_instance = applications.Slicer2DPlotter(vol)
        vedo.plotter_instance.show().close()
        return

    ########################################################################
    # normal mode for single VOXEL file with Isosurface Slider mode
    elif nfiles == 1 and (
        ".slc" in args.files[0].lower()
        or ".vti" in args.files[0].lower()
        or ".tif" in args.files[0].lower()
        or ".mhd" in args.files[0].lower()
        or ".nrrd" in args.files[0].lower()
        or ".dem" in args.files[0].lower()
    ):
        # print('DEBUG normal mode for single VOXEL file with Isosurface Slider mode')
        vol = file_io.load(args.files[0], force=args.reload)
        sp = vol.spacing()
        vol.spacing([sp[0] * args.x_spacing, 
                     sp[1] * args.y_spacing, 
                     sp[2] * args.z_spacing])
        if not args.color:
            args.color = "gold"
        plt = applications.IsosurfaceBrowser(
            vol, c=args.color, cmap=args.cmap, precompute=False, use_gpu=True
        )
        plt.show(zoom=args.zoom, viewup="z")
        return

    ########################################################################
    # NORMAL mode for single or multiple files, or multiren mode, or numpy scene
    elif nfiles == 1 or (not args.sequence_mode):
        # print('DEBUG NORMAL mode for single or multiple files, or multiren mode')

        interactor_mode = 0
        if args.image:
            interactor_mode = "image"

        ##########################################################
        # loading a full scene or list of objects
        if ".npy" in args.files[0] or ".npz" in args.files[0]:
            try: # full scene
                plt = file_io.import_window(args.files[0])
                plt.show(mode=interactor_mode).close()
                return
            except KeyError: # list of objects, create Assembly
                objs = vedo.Assembly(args.files[0])
                for i, ob in enumerate(objs):
                    if ob:
                        ob.c(i)
                plt = Plotter()
                plt.show(objs, mode=interactor_mode).close()
                return
        #########################################################

        ds = 0
        objs = []

        for i in range(N):
            f = args.files[i]

            colb = args.color
            if args.color is None and N > 1:
                colb = i

            obj = load(f, force=args.reload)

            if isinstance(obj, (TetMesh, UnstructuredGrid)):
                # obj = obj#.shrink(0.95)
                obj.c(colb).alpha(args.alpha)

            elif isinstance(obj, vedo.Points):
                obj.c(colb).alpha(args.alpha)

                try:
                    obj.wireframe(args.wireframe)
                    if args.flat:
                        obj.flat()
                    else:
                        obj.phong()
                except AttributeError:
                    pass

                obj.lighting(args.lighting)

                if i == 0 and args.texture_file:
                    obj.texture(args.texture_file)

                if args.point_size > 0:
                    obj.ps(args.point_size)

                if args.cmap != "jet":
                    obj.cmap(args.cmap)

                if args.showedges:
                    try:
                        obj.GetProperty().SetEdgeVisibility(1)
                        obj.GetProperty().SetLineWidth(0.1)
                        obj.GetProperty().SetRepresentationToSurface()
                    except AttributeError:
                        pass

            objs.append(obj)

            if args.multirenderer_mode:
                try:
                    ds = obj.diagonal_size() * 3
                    plt.camera.SetClippingRange(0, ds)
                    plt.reset_camera()
                    # plt.render()
                    plt.show(
                        obj,
                        at=i,
                        interactive=False,
                        zoom=args.zoom,
                        mode=interactor_mode,
                    )

                except AttributeError:
                    # wildcards in quotes make glob return obj as a list :(
                    vedo.logger.error("Please do not use wildcards within single or double quotes")

        if args.multirenderer_mode:
            plt.interactor.Start()
            if vedo.vtk_version == (9, 2, 2):
                plt.interactor.GetRenderWindow().SetDisplayId("_0_p_void")

        else:

            # scene is empty
            if all(a is None for a in objs):
                vedo.logger.error("Could not load file(s). Quit.")
                return
            plt.show(objs, interactive=True, zoom=args.zoom, mode=interactor_mode)
        return

    ########################################################################
    # sequence mode  -s
    else:
        # print("DEBUG simple browser mode  -s")
        if plt.axes == 4:
            plt.axes = 1

        acts = load(args.files, force=args.reload)
        plt += acts
        for a in acts:
            if hasattr(a, "c"):  # Image doesnt have it
                a.c(args.color)

            if args.point_size > 0:
                try:
                    a.GetProperty().SetPointSize(args.point_size)
                    a.GetProperty().SetRepresentationToPoints()
                except AttributeError:
                    pass

            if args.cmap != "jet":
                try:
                    a.cmap(args.cmap)
                except AttributeError:
                    pass

            try:
                a.lighting(args.lighting)
            except AttributeError:
                pass

            a.alpha(args.alpha)

        plt = applications.Browser(acts, axes=1)
        plt.show(zoom=args.zoom).close()
