#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

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
import importlib
import inspect
import os
import pkgutil
import re
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version as pkg_version

__all__ = []

_IMAGE_OUTPUT_EXTS = {".png", ".jpg", ".jpeg", ".pdf", ".svg", ".eps"}
_SCENE_OUTPUT_EXTS = {".npy", ".npz", ".x3d", ".html"}

vedo = None
np = None
humansort = None
get_color = None
printc = None


def _get_pkg_version(dist_name: str, fallback: str = "unknown") -> str:
    """Return the installed distribution version or a fallback label."""
    try:
        return pkg_version(dist_name)
    except PackageNotFoundError:
        return fallback


def _get_install_dir() -> str:
    """Return the package directory, or the project root in a source checkout."""
    package_dir = os.path.dirname(os.path.realpath(__file__))
    project_root = os.path.dirname(package_dir)

    if (
        os.path.isfile(os.path.join(package_dir, "__init__.py"))
        and os.path.isfile(os.path.join(project_root, "pyproject.toml"))
        and os.path.isdir(os.path.join(project_root, "examples"))
    ):
        return project_root
    return package_dir


def _get_gpu_info_rows() -> list[tuple[str, str]]:
    """Return GPU information rows for system_info(), when available."""
    try:
        from vtkmodules.vtkCommonCore import vtkObject
        from vtkmodules.vtkRenderingOpenGL2 import vtkOpenGLRenderWindow

        previous_warning_state = vtkObject.GetGlobalWarningDisplay()
        render_window = None
        try:
            vtkObject.GlobalWarningDisplayOff()
            render_window = vtkOpenGLRenderWindow()
            render_window.SetOffScreenRendering(1)
            render_window.Initialize()
            capabilities = render_window.ReportCapabilities() or ""
        finally:
            try:
                if render_window is not None:
                    render_window.Finalize()
            except Exception:
                pass
            if previous_warning_state:
                vtkObject.GlobalWarningDisplayOn()
            else:
                vtkObject.GlobalWarningDisplayOff()
    except Exception:
        return []

    prefixes = {
        "OpenGL vendor string": "vendor",
        "OpenGL renderer string": "renderer",
        "OpenGL version string": "version",
    }
    values = {}
    for line in capabilities.splitlines():
        for prefix, key in prefixes.items():
            if line.startswith(prefix):
                values[key] = line.split(":", 1)[1].strip()
                break

    rows = []
    renderer = values.get("renderer")
    vendor = values.get("vendor")
    if renderer and vendor:
        rows.append(("GPU Renderer", f"{vendor} {renderer}"))
    elif renderer:
        rows.append(("GPU Renderer", renderer))
    elif vendor:
        rows.append(("GPU Vendor", vendor))

    version = values.get("version")
    if version:
        rows.append(("GPU Version", version))
    return rows


def _ensure_cli_runtime():
    """Import and cache CLI runtime dependencies, then return the vedo module."""
    global vedo, np, humansort, get_color, printc
    install_dir = _get_install_dir()
    if vedo is not None:
        vedo.installdir = install_dir
        return vedo

    import numpy as np_module
    import vedo as vedo_module
    from vedo.colors import get_color as get_color_fn, printc as printc_fn
    from vedo.utils import humansort as humansort_fn

    vedo = vedo_module
    np = np_module
    humansort = humansort_fn
    get_color = get_color_fn
    printc = printc_fn
    vedo.installdir = install_dir
    return vedo


def _normalize_output_path(output_path: str | None) -> str:
    """Return a normalized CLI output path."""
    return (output_path or "").strip()


def _classify_output_path(output_path: str | None) -> str | None:
    """Classify an output path as an image export, scene export, or unsupported."""
    normalized = _normalize_output_path(output_path)
    if not normalized:
        return None

    ext = os.path.splitext(normalized)[1].lower()
    if ext in _IMAGE_OUTPUT_EXTS:
        return "image"
    if ext in _SCENE_OUTPUT_EXTS:
        return "scene"
    return None


def _should_render_offscreen(args) -> bool:
    """Return whether the CLI should render non-interactively."""
    return bool(args.offscreen or _normalize_output_path(args.output))


def _write_cli_output(plt, args) -> bool:
    """Write the current plotter output if requested."""
    output_path = _normalize_output_path(args.output)
    if not output_path:
        return False

    output_kind = _classify_output_path(output_path)
    if output_kind == "image":
        plt.screenshot(output_path, scale=args.scale)
    elif output_kind == "scene":
        vedo.file_io.export_window(output_path, plt=plt, backend=args.backend)
    else:
        raise ValueError(f"Unsupported output path: {output_path}")

    vedo.logger.info(f"Saved output to {output_path}")
    return True


def _show_and_finalize(plt, args, *objects, close_on_exit=False, **show_kwargs):
    """Show a plotter, optionally export the result, and close when appropriate."""
    if _should_render_offscreen(args):
        show_kwargs["interactive"] = False

    plt.show(*objects, **show_kwargs)

    if args.output:
        _write_cli_output(plt, args)

    if args.output or args.offscreen or close_on_exit:
        plt.close()
    return plt


##############################################################################################
def main():
    """Execute the command line interface and return the result."""
    parser = get_parser()
    args = parser.parse_args()
    args.output = _normalize_output_path(args.output)

    if args.scale < 1:
        parser.error("--scale must be greater than or equal to 1")

    output_kind = _classify_output_path(args.output)
    if args.output and output_kind is None:
        parser.error(
            "--output format must be one of: "
            + ", ".join(sorted(_IMAGE_OUTPUT_EXTS | _SCENE_OUTPUT_EXTS))
        )

    if args.backend and output_kind != "scene":
        parser.error("--backend can only be used with --output <file.html>")

    if args.backend and not args.output.lower().endswith(".html"):
        parser.error("--backend can only be used with HTML scene exports")

    if args.info:
        system_info()
        return 0

    elif args.run:
        _ensure_cli_runtime()
        exe_run(args)
        return 0

    elif args.search:
        _ensure_cli_runtime()
        exe_search(args)
        return 0

    elif args.search_vtk:
        exe_search_vtk(args)
        return 0

    elif args.search_code:
        _ensure_cli_runtime()
        exe_search_code(args)
        return 0

    elif args.locate:
        _ensure_cli_runtime()
        exe_locate(args)
        return 0

    elif args.convert is not None:
        if not args.convert:
            parser.error("--convert requires at least one input file")
        _ensure_cli_runtime()
        exe_convert(args)
        return 0

    elif args.eog:
        _ensure_cli_runtime()
        exe_eog(args)
        return 0

    elif not args.files:
        system_info()
        message = "No input files provided. Try one of these:"
        example = (
            "vedo https://vedo.embl.es/examples/data/panther.stl\n"
            "or explore a built-in example with:\n"
            "vedo -r warp1"
        )
        try:
            from rich import box
            from rich.console import Console
            from rich.panel import Panel
            from rich.text import Text

            body = Text()
            body.append(f"{message}\n", style="bold yellow")
            body.append("vedo https://vedo.embl.es/examples/data/panther.stl\n", style="bold white")
            body.append("or explore a built-in example with:\n", style="bold yellow")
            body.append("vedo -r warp1", style="bold white")
            Console().print(
                Panel(
                    body,
                    box=box.ROUNDED,
                    title_align="left",
                    border_style="yellow",
                    padding=(0, 1),
                    expand=False,
                )
            )
        except ModuleNotFoundError:
            print(f":idea: {message}")
            print(f" {example}")
        return 0

    else:
        _ensure_cli_runtime()
        draw_scene(args)
        return 0


##############################################################################################
def get_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""

    descr = f"version {_get_pkg_version('vedo')}"
    descr += " - check out home page at https://vedo.embl.es"

    pr = argparse.ArgumentParser(
        description=descr,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    action_group = pr.add_mutually_exclusive_group()
    pr.add_argument("files", nargs="*", help="input filename(s)")
    pr.add_argument(
        "-c",
        "--color",
        type=str,
        help="mesh color [integer or color name]",
        default=None,
        metavar="",
    )
    pr.add_argument(
        "-a", "--alpha", type=float, help="alpha value [0-1]", default=1, metavar=""
    )
    pr.add_argument(
        "-w", "--wireframe", help="use wireframe representation", action="store_true"
    )
    pr.add_argument(
        "-p",
        "--point-size",
        type=float,
        help="specify point size",
        default=-1,
        metavar="",
    )
    pr.add_argument(
        "-l", "--showedges", help="show a thin line on mesh edges", action="store_true"
    )
    pr.add_argument(
        "-k",
        "--lighting",
        type=str,
        help="metallic, plastic, shiny, glossy or off",
        default="default",
        metavar="",
    )
    pr.add_argument("-K", "--flat", help="use flat shading", action="store_true")
    pr.add_argument(
        "-t", "--texture-file", help="texture image file", default="", metavar=""
    )
    pr.add_argument(
        "-x",
        "--axes-type",
        type=int,
        help="specify axes type [0-14]",
        default=1,
        metavar="",
    )
    pr.add_argument(
        "-i",
        "--no-camera-share",
        help="do not share camera in renderers",
        action="store_true",
    )
    pr.add_argument("-f", "--full-screen", help="full screen mode", action="store_true")
    pr.add_argument(
        "-bg",
        "--background",
        type=str,
        help="background color [integer or color name]",
        default="",
        metavar="",
    )
    pr.add_argument(
        "-bg2",
        "--background-grad",
        help="use background color gradient",
        default="",
        metavar="",
    )
    pr.add_argument(
        "-z", "--zoom", type=float, help="zooming factor", default=1, metavar=""
    )
    pr.add_argument(
        "-n",
        "--multirenderer-mode",
        help="multi renderer mode: files go to separate renderers",
        action="store_true",
    )
    pr.add_argument(
        "-s",
        "--sequence-mode",
        help="sequence mode: use slider to browse files",
        action="store_true",
    )
    pr.add_argument(
        "-g",
        "--ray-cast-mode",
        help="GPU Ray-casting Mode for 3D image files",
        action="store_true",
    )
    pr.add_argument(
        "-gx",
        "--x-spacing",
        type=float,
        help="volume x-spacing factor [1]",
        default=1,
        metavar="",
    )
    pr.add_argument(
        "-gy",
        "--y-spacing",
        type=float,
        help="volume y-spacing factor [1]",
        default=1,
        metavar="",
    )
    pr.add_argument(
        "-gz",
        "--z-spacing",
        type=float,
        help="volume z-spacing factor [1]",
        default=1,
        metavar="",
    )
    pr.add_argument(
        "--mode",
        type=int,
        help="volume rendering mode [0=composite, 1=maxproj, 2=minproj, 3=avg, 4=additive]",
        default=0,
        metavar="",
    )
    pr.add_argument(
        "--cmap", help="volume rendering color map name", default="jet", metavar=""
    )
    pr.add_argument(
        "-e", "--edit", help="free-hand edit the input Mesh", action="store_true"
    )
    pr.add_argument(
        "--slicer2d", help="2D Slicer Mode for volumetric data", action="store_true"
    )
    pr.add_argument(
        "--slicer3d", help="3D Slicer Mode for volumetric data", action="store_true"
    )
    action_group.add_argument("-r", "--run", help="run example from vedo/examples", metavar="")
    action_group.add_argument(
        "--search",
        type=str,
        help="search/grep for word in vedo examples",
        default="",
        metavar="",
    )
    action_group.add_argument(
        "--search-vtk",
        type=str,
        help="search examples for the input vtk class",
        default="",
        metavar="",
    )
    action_group.add_argument(
        "--search-code",
        type=str,
        help="search keyword in source code",
        default="",
        metavar="",
    )
    action_group.add_argument(
        "--locate",
        type=str,
        help="locate module path of a vedo class",
        default="",
        metavar="",
    )
    pr.add_argument(
        "--reload",
        help="reload the file, ignoring any previous download",
        action="store_true",
    )
    action_group.add_argument(
        "--info", help="get an info printout of the current installation", action="store_true"
    )
    action_group.add_argument("--convert", nargs="+", help="input file(s) to be converted")
    pr.add_argument(
        "--to",
        type=str,
        help="convert to this target format",
        default="vtk",
        metavar="",
    )
    pr.add_argument("--image", help="image mode for 2d objects", action="store_true")
    action_group.add_argument("--eog", help="eog-like image visualizer", action="store_true")
    pr.add_argument("--font", help="font name", default="Normografo", metavar="")
    pr.add_argument(
        "--output",
        type=str,
        help="write a non-interactive output (.png, .jpg, .pdf, .svg, .eps, .html, .x3d, .npy, .npz)",
        default="",
        metavar="",
    )
    pr.add_argument(
        "--backend",
        choices=("k3d", "threejs"),
        help="scene export backend for HTML outputs",
        default=None,
        metavar="",
    )
    pr.add_argument(
        "--offscreen",
        help="render without opening a window",
        action="store_true",
    )
    pr.add_argument(
        "--scale",
        type=int,
        help="screenshot magnification factor",
        default=1,
        metavar="",
    )
    return pr


#################################################################################################
def system_info() -> None:
    """Print a summary of the current vedo runtime environment."""
    vedo_version = _get_pkg_version("vedo")
    vtk_version = _get_pkg_version("vtk", fallback="unknown")
    numpy_version = _get_pkg_version("numpy", fallback="unknown")
    rows = [
        ("vedo version", vedo_version),
        ("homepage", "https://vedo.embl.es"),
        ("vtk version", vtk_version),
        ("numpy version", numpy_version),
        ("python version", " ".join(sys.version.split())),
        ("python interpreter", sys.executable),
        ("installation point", _get_install_dir()),
    ]

    try:
        import platform

        rows.append(
            (
                "system",
                " ".join(
                    [
                        platform.system(),
                        platform.release(),
                        os.name,
                        platform.machine(),
                    ]
                ),
            )
        )
    except ModuleNotFoundError:
        pass

    rows.extend(_get_gpu_info_rows())

    # k3d_version = _get_pkg_version("k3d", fallback=None)
    # if k3d_version is not None:
    #     rows.append(("k3d version", k3d_version))

    try:
        from rich import box
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table
        from rich.text import Text

        table = Table(
            show_header=False,
            box=None,
            expand=False,
            pad_edge=False,
            padding=(0, 1),
        )
        table.add_column(style="bold bright_cyan", no_wrap=True)
        table.add_column(style="white")
        for key, value in rows:
            table.add_row(key, value)
        Console().print(
            Panel(
                table,
                box=box.ROUNDED,
                # title=Text("vedo", style="bold white"),
                subtitle=Text("System Info", style="bold cyan"),
                title_align="left",
                subtitle_align="right",
                border_style="bright_blue",
                padding=(0, 1),
                expand=False,
            )
        )
    except ModuleNotFoundError:
        for key, value in rows:
            print(f"{key + ':':18} {value}")

    # try:
    #     import trame
    #     printc("trame version     :", trame.__version__, bold=0, dim=1)
    # except ModuleNotFoundError:
    #     pass


#################################################################################################
def exe_run(args) -> None:
    """Run an example script, or list matching examples when the query is ambiguous."""
    expath = os.path.join(vedo.installdir, "examples", "**", "*.py")
    exfiles = list(glob.glob(expath, recursive=True))
    f2search = os.path.basename(args.run).lower()
    matching = [
        s
        for s in exfiles
        if (
            f2search in os.path.basename(s).lower()
            and "__" not in s
        )
    ]
    humansort(matching)
    nmat = len(matching)
    if nmat == 0:
        vedo.logger.warning(f"No matching example with name: {args.run}")
        # Nothing found, try to search for a script content:
        search_args = argparse.Namespace(**vars(args))
        search_args.search = args.run
        search_args.run = ""
        exe_search(search_args)
        return

    show_summary = args.full_screen or nmat > 1
    if nmat > 1:
        printc(f":target: Found {nmat} scripts containing string '{args.run}':", c="c")

    if show_summary:  # -f option not to dump the full code but just the first line
        for mat in matching[:30]:
            printc(os.path.basename(mat).replace(".py", ""), c="c", end=" ")
            with open(mat, "r", encoding="UTF-8") as fm:
                lline = "".join(line for _, line in zip(range(60), fm))
                maxidx1 = lline.find("import ")
                maxidx2 = lline.find("from vedo")
                cut_points = [idx for idx in (maxidx1, maxidx2) if idx >= 0]
                if cut_points:
                    lline = lline[: min(cut_points)]  # cut where the code starts
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
        printc(f"... (and {nmat - 30} more)", c="c")

    if nmat > 1:
        printc(":idea: Type 'vedo -r <name>' to run one of them", bold=0, c="c")
        return

    if not show_summary:  # -f option not to dump the full code
        with open(matching[0], "r", encoding="UTF-8") as fm:
            code = fm.read()
        code = "#" * 80 + "\n" + code + "\n" + "#" * 80

        from pygments import highlight
        from pygments.lexers import Python3Lexer
        from pygments.formatters import Terminal256Formatter
        from pygments.styles import STYLE_MAP

        # print("Terminal256Formatter STYLE_MAP", STYLE_MAP.keys())
        if "zenburn" in STYLE_MAP.keys():
            tform = Terminal256Formatter(style="zenburn")
        elif "monokai" in STYLE_MAP.keys():
            tform = Terminal256Formatter(style="monokai")
        else:
            tform = Terminal256Formatter()
        result = highlight(code, Python3Lexer(), tform)
        print(result, end="")

    printc("(" + matching[0] + ")", c="y", bold=0, italic=1)
    cmd = [sys.executable, matching[0]]
    try:
        subprocess.run(cmd, check=False)
    except OSError:
        # Last-resort fallback for unusual interpreter setups.
        subprocess.run(["python3", matching[0]], check=False)


################################################################################################
def exe_convert(args) -> None:
    """Convert input files to a different supported output format."""

    allowed_exts = [
        "vtk",
        "vtp",
        "vtu",
        "vts",
        "npy",
        "ply",
        "stl",
        "obj",
        "off",
        "byu",
        "vti",
        "tif",
        "mhd",
    ]

    humansort(args.convert)
    nfiles = len(args.convert)
    if nfiles == 0:
        return

    target_ext = args.to.lower()

    if target_ext not in allowed_exts:
        vedo.logger.error(
            f"Sorry target cannot be {target_ext}. Must be {allowed_exts}"
        )
        sys.exit()

    for f in args.convert:
        root, source_ext = os.path.splitext(f)
        source_ext = source_ext.lower().lstrip(".")

        if not source_ext:
            vedo.logger.warning(f"Skipping {f}: cannot determine source file extension")
            continue

        if target_ext == source_ext:
            continue

        a = vedo.load(f)
        newf = root + "." + target_ext
        a.write(newf, binary=True)


##############################################################################################
def exe_search(args) -> None:
    """Search the example scripts for a text pattern and print matching lines."""
    expath = os.path.join(vedo.installdir, "examples", "**", "*.py")
    exfiles = list(glob.glob(expath, recursive=True))
    humansort(exfiles)
    pattern = args.search
    ignore_case = args.no_camera_share
    if len(pattern) <= 3:
        vedo.logger.warning("Please use at least 4 letters in keyword search!")
        return

    flags = re.IGNORECASE if ignore_case else 0
    matcher = re.compile(re.escape(pattern), flags)
    examples_dir = os.path.join(vedo.installdir, "examples")
    highlight_prefix = "\x1b[4m\x1b[1m"
    highlight_suffix = "\x1b[0m\u001b[33m"

    for ifile in exfiles:
        with open(ifile, "r", encoding="UTF-8") as file:
            show_file_header = True
            for line_no, line in enumerate(file, start=1):
                if not matcher.search(line):
                    continue
                if show_file_header:
                    relpath = os.path.relpath(ifile, examples_dir)
                    printc(
                        f"--> examples/{relpath}:",
                        c="y",
                        italic=1,
                        invert=1,
                    )
                    show_file_header = False
                highlighted = matcher.sub(
                    lambda match: f"{highlight_prefix}{match.group(0)}{highlight_suffix}",
                    line,
                )
                print(f"\u001b[33m{line_no}\t{highlighted}\x1b[0m", end="")
                # printc(line_no, highlighted, c='y', bold=False, end='')


##############################################################################################
def exe_search_code(args) -> None:
    """Search vedo source code for a keyword and print matching symbols."""

    from pygments import highlight
    from pygments.lexers import Python3Lexer
    from pygments.formatters import Terminal256Formatter

    # styles: autumn, material, rrt, zenburn
    style = "zenburn"
    key = (args.search_code or "").strip()
    iopt = args.no_camera_share
    if key.lower() == key:
        iopt = True

    if len(key) < 4:
        vedo.logger.warning("Please use at least 4 letters in keyword search!")
        return

    matcher = re.compile(re.escape(key), re.IGNORECASE if iopt else 0)

    def _dump(mcontent) -> None:
        for name, member in mcontent:
            if name.startswith("_"):
                continue
            if name.startswith("vtk"):
                continue

            try:
                source = inspect.getsource(member)
            except (OSError, TypeError):
                continue

            if source is None:
                continue

            if "eprecated" in source.lower():
                continue

            if not matcher.search(name) and not matcher.search(source):
                continue

            module = inspect.getmodule(member)
            module_name = module.__name__ if module else "<unknown>"
            sname = module_name + " -> " + name
            if sname in snames:
                continue
            snames.add(sname)

            try:
                filename = os.path.basename(inspect.getfile(member))
            except (OSError, TypeError):
                filename = "<unknown>"

            printc(
                ":checked:Found matching",
                sname,
                "in module",
                filename,
                c="y",
                invert=True,
            )
            source = source.replace("``", '"').replace("`", '"')
            source = source.replace(".. warning::", "Warning!")
            result = highlight(source, Python3Lexer(), Terminal256Formatter(style=style))
            print(result, end="")
            print("\x1b[0m")

    # printc("..parsing source code, please wait", c="y", bold=False)
    content = inspect.getmembers(vedo)
    snames: set[str] = set()
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
        if inspect.isclass(m):
            _dump(inspect.getmembers(m))


##############################################################################################
def exe_search_vtk(args) -> None:
    """Search the VTK examples cross-reference for a class name."""

    # input a vtk class name to get links to examples that involve that class
    # From https://kitware.github.io/vtk-examples/site/Python/Utilities/SelectExamples/
    import json
    import tempfile
    import time
    from pathlib import Path
    from urllib.error import HTTPError, URLError
    from urllib.request import urlretrieve

    xref_url = "https://raw.githubusercontent.com/Kitware/vtk-examples/gh-pages/src/Coverage/vtk_vtk-examples_xref.json"

    vtk_class = (args.search_vtk or "").strip()
    if not vtk_class:
        vedo.logger.warning("Please provide a VTK class name, e.g. `vedo --search-vtk vtkActor`")
        return

    def _download_file(dl_path: str, dl_url: str, overwrite: bool = False) -> Path:
        file_name = dl_url.split("/")[-1]
        # Create necessary sub-directories in the dl_path (if they don't exist).
        Path(dl_path).mkdir(parents=True, exist_ok=True)
        # Download if it doesn't exist in the directory overriding if overwrite is True.
        path = Path(dl_path, file_name)
        if not path.is_file() or overwrite:
            try:
                urlretrieve(dl_url, path)
            except (HTTPError, URLError, OSError) as e:
                reason = getattr(e, "reason", str(e))
                raise RuntimeError(f"Failed to download {dl_url}: {reason}") from e
        return path

    def _get_examples(data: dict, class_name: str, lang: str) -> list[str]:
        class_examples = data.get(class_name, {})
        if not isinstance(class_examples, dict):
            return []
        language_examples = class_examples.get(lang, {})
        if not isinstance(language_examples, dict):
            return []
        return [str(example) for example in language_examples.values()]

    language = "Python"
    tmp_dir = tempfile.gettempdir()
    try:
        path = _download_file(tmp_dir, xref_url, overwrite=False)
    except RuntimeError as exc:
        vedo.logger.warning(str(exc))
        return

    if not path.is_file():
        vedo.logger.warning(f"VTK examples index not found at {path}")
        return

    cache_age = time.time() - path.stat().st_mtime
    # Force a new download if the time difference is > 10 minutes.
    if cache_age > 600:
        try:
            path = _download_file(tmp_dir, xref_url, overwrite=True)
        except RuntimeError as exc:
            vedo.logger.warning(f"{exc}. Reusing cached index at {path}.")

    try:
        with open(path, "r", encoding="UTF-8") as json_file:
            xref_dict = json.load(json_file)
    except (OSError, json.JSONDecodeError) as exc:
        vedo.logger.warning(f"Failed to read VTK examples index {path}: {exc}")
        return

    examples = _get_examples(xref_dict, vtk_class, language)
    if examples:
        print(
            f"VTK Class: {vtk_class}, language: {language}\n"
            f"Number of example(s): {len(examples)}."
        )
        print("\n".join(examples))
    else:
        print(f"No examples for the VTK Class: {vtk_class} and language: {language}")


##############################################################################################
def exe_locate(args) -> None:
    """Locate the fully qualified module path for a vedo class name."""
    target = (args.locate or "").strip()
    if not target:
        vedo.logger.warning(
            "Please provide a class name, e.g. `vedo --locate Paraboloid`"
        )
        return

    target_lower = target.lower()
    matches: set[str] = set()
    case_insensitive_matches: set[str] = set()
    class_names_seen: set[str] = set()

    # Fast path: check top-level vedo namespace first.
    try:
        obj = getattr(vedo, target)
    except AttributeError:
        obj = None
    if inspect.isclass(obj):
        matches.add(f"{obj.__module__}.{obj.__name__}")

    for name, obj in inspect.getmembers(vedo, inspect.isclass):
        class_names_seen.add(name)
        if name.lower() == target_lower:
            case_insensitive_matches.add(f"{obj.__module__}.{obj.__name__}")

    skip_prefixes = ("vedo.backends",)
    for module_info in pkgutil.walk_packages(vedo.__path__, prefix="vedo."):
        module_name = module_info.name
        if module_name.startswith(skip_prefixes):
            continue
        try:
            module = importlib.import_module(module_name)
        except Exception:
            # Optional dependencies or side effects may make some modules unavailable.
            continue

        for cname, cls in inspect.getmembers(module, inspect.isclass):
            if cls.__module__ != module_name:
                continue
            class_names_seen.add(cname)
            if cname == target:
                matches.add(f"{module_name}.{cname}")
            if cname.lower() == target_lower:
                case_insensitive_matches.add(f"{module_name}.{cname}")

    if matches:
        for match in sorted(matches):
            print(match)
        return

    # Case-insensitive fallback and friendly hint.
    fuzzy = sorted(name for name in class_names_seen if target_lower in name.lower())[
        :20
    ]
    if case_insensitive_matches:
        for match in sorted(case_insensitive_matches):
            print(match)
        return

    vedo.logger.warning(f"No vedo class found with name '{target}'.")
    if fuzzy:
        printc(":idea: Similar class names:", c="c")
        for name in fuzzy:
            print(" ", name)


#################################################################################################################
def exe_eog(args) -> None:
    """Display one or more images with the lightweight EOG-style viewer."""
    settings = vedo.settings
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

    files = [f for f in args.files if not f.lower().endswith(".gif")]

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
            elif event.keypress == "Left":
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
                vedo.logger.info("Image saved as screenshot.png")
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
                pic = vedo.Image(f)
                if pic:
                    pics.append(pic)
            except Exception:
                vedo.logger.error(f"Could not load image {f}")
        else:
            vedo.logger.error(f"Could not load image {f}")

    n = len(pics)
    if not n:
        vedo.logger.warning("No loadable image files were provided to --eog.")
        return

    render_offscreen = _should_render_offscreen(args)
    pic = pics[0]
    lev, win = pic.level(), pic.window()

    if n > 1:
        plt = vedo.Plotter(
            N=n,
            sharecam=True,
            bg=args.background,
            bg2=args.background_grad,
            offscreen=render_offscreen,
        )
        plt.add_callback("key press", vfunc)
        for i in range(n):
            p = pics[i].pickable(True)
            pos = [-p.shape[0] / 2, -p.shape[1] / 2, 0]
            p.pos(pos)
            plt.add_hover_legend(at=i, c="k8", bg="k2", alpha=0.4)
            plt.show(p, axes=0, at=i, mode="image")
        plt.show(interactive=False)
        plt.reset_camera(tight=0.05)
        if args.output:
            _write_cli_output(plt, args)
            plt.close()
            return
        if args.offscreen:
            plt.close()
            return
        plt.interactor.Start()
        if vedo.vtk_version == (9, 2, 2):
            plt.interactor.GetRenderWindow().SetDisplayId("_0_p_void")

    else:
        width, height = pic.shape[:2]
        if width > 1500:
            height = height / width * 1500
            width = 1500

        if height > 1200:
            width = width / height * 1200
            height = 1200

        size = (int(round(width)), int(round(height)))

        plt = vedo.Plotter(
            title=files[0],
            size=size,
            bg=args.background,
            bg2=args.background_grad,
            offscreen=render_offscreen,
        )
        plt.add_callback("key press", vfunc)
        plt.add_hover_legend(c="k8", bg="k2", alpha=0.4)
        plt.show(pic, mode="image", interactive=False)
        plt.reset_camera(tight=0.0)
        if args.output:
            _write_cli_output(plt, args)
            plt.close()
            return
        if args.offscreen:
            plt.close()
            return
        plt.interactor.Start()
        if vedo.vtk_version == (9, 2, 2):
            plt.interactor.GetRenderWindow().SetDisplayId("_0_p_void")

    plt.close()


#################################################################################################################
def draw_scene(args) -> None:
    """Load the requested input files and dispatch them to the appropriate viewer."""
    settings = vedo.settings
    if settings.dry_run_mode >= 2:
        print(f"draw_scene called in dry run mode {settings.dry_run_mode}. Skip.")
        return

    nfiles = len(args.files)
    if nfiles == 0:
        vedo.logger.error("No input files.")
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

    first_file = args.files[0]
    first_file_lower = first_file.lower()

    if nfiles == 1 and first_file_lower.endswith(".gif"):
        frames = vedo.load(args.files[0])
        plt = vedo.applications.Browser(
            frames,
            offscreen=_should_render_offscreen(args),
        )
        _show_and_finalize(
            plt,
            args,
            bg=args.background,
            bg2=args.background_grad,
            close_on_exit=True,
        )
        return  ##########################################################

    multirenderer_mode = args.multirenderer_mode and not args.sequence_mode
    render_offscreen = _should_render_offscreen(args)
    settings.default_font = args.font

    sharecam = not args.no_camera_share

    N = None
    if multirenderer_mode:
        if nfiles < 201:
            N = nfiles
        if nfiles > 200:
            vedo.logger.warning("Option '-n' allows a maximum of 200 files.")
            vedo.logger.warning(f"You are trying to load {nfiles} files.")
            N = 200
        if N > 4:
            settings.use_depth_peeling = False

    else:
        N = nfiles

    ##########################################################
    # special case of SLC/TIFF volumes with -g option
    if args.ray_cast_mode:
        # print('DEBUG special case of SLC/TIFF volumes with -g option')

        vol = vedo.file_io.load(args.files[0], force=args.reload)

        if not isinstance(vol, vedo.Volume):
            vedo.logger.error(f"expected a Volume but loaded a {type(vol)} object")
            return

        sp = vol.spacing()
        vol.spacing(
            [sp[0] * args.x_spacing, sp[1] * args.y_spacing, sp[2] * args.z_spacing]
        )
        vol.mode(int(args.mode)).color(args.cmap).jittering(True)
        plt = vedo.applications.RayCastPlotter(vol, offscreen=render_offscreen)
        _show_and_finalize(plt, args, viewup="z", close_on_exit=True)
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

        vol = vedo.file_io.load(args.files[0], force=args.reload)

        sp = vol.spacing()
        vol.spacing(
            [sp[0] * args.x_spacing, sp[1] * args.y_spacing, sp[2] * args.z_spacing]
        )

        vedo.set_current_plotter(None)  # reset

        plt = vedo.applications.Slicer3DPlotter(
            vol,
            bg="white",
            bg2="lb",
            use_slider3d=useSlider3D,
            axes=args.axes_type,
            clamp=True,
            size=(1350, 1000),
            offscreen=render_offscreen,
        )
        plt += vedo.Text2D(args.files[0], pos="top-left", font="VictorMono", s=1, c="k")
        _show_and_finalize(plt, args, close_on_exit=True)
        return

    ########################################################################
    elif args.edit:
        # print('edit mode for meshes and pointclouds')
        if render_offscreen:
            vedo.logger.warning(
                "Option '--edit' is interactive-only and cannot be combined with "
                "--output or --offscreen."
            )
            return
        vedo.set_current_plotter(None)  # reset
        settings.use_parallel_projection = True

        try:
            m = vedo.Mesh(args.files[0], alpha=args.alpha / 2, c=args.color)
        except AttributeError:
            vedo.logger.critical(
                "In edit mode, input file must be a point cloud or polygonal mesh."
            )
            return

        plt = vedo.applications.FreeHandCutPlotter(m, splined=True)
        plt.add_hover_legend()
        if not args.background_grad:
            args.background_grad = None
        plt.start(axes=1, bg=args.background, bg2=args.background_grad)

    ########################################################################
    elif args.slicer2d:
        # print('DEBUG special case of SLC/TIFF/DICOM volumes with --slicer2d option')
        vol = vedo.file_io.load(args.files[0], force=args.reload)
        if not vol:
            return
        vol.cmap("bone_r")
        sp = vol.spacing()
        vol.spacing(
            [sp[0] * args.x_spacing, sp[1] * args.y_spacing, sp[2] * args.z_spacing]
        )
        plt = vedo.set_current_plotter(
            vedo.applications.Slicer2DPlotter(vol, offscreen=render_offscreen)
        )
        _show_and_finalize(plt, args, close_on_exit=True)
        return

    ########################################################################
    # normal mode for single VOXEL file with Isosurface Slider mode
    elif nfiles == 1 and (
        first_file_lower.endswith((".slc", ".vti", ".tif", ".mhd", ".nrrd", ".dem", ".dx"))
    ):
        # print('DEBUG normal mode for single VOXEL file with Isosurface Slider mode')
        vol = vedo.file_io.load(args.files[0], force=args.reload)

        if vol.shape[2] == 1:
            # print('DEBUG It is a 2D image!')
            img = vedo.Image(args.files[0])
            plt = vedo.Plotter(offscreen=render_offscreen).parallel_projection()
            _show_and_finalize(
                plt,
                args,
                img,
                zoom="tightest",
                mode="image",
                close_on_exit=True,
            )
            return

        sp = vol.spacing()
        vol.spacing(
            [sp[0] * args.x_spacing, sp[1] * args.y_spacing, sp[2] * args.z_spacing]
        )
        if not args.color:
            args.color = "gold"
        plt = vedo.applications.IsosurfaceBrowser(
            vol,
            c=args.color,
            cmap=args.cmap,
            precompute=False,
            use_gpu=True,
            offscreen=render_offscreen,
        )
        _show_and_finalize(plt, args, zoom=args.zoom, viewup="z", close_on_exit=True)
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
        if first_file_lower.endswith((".npy", ".npz")):
            try:  # full scene
                plt = vedo.file_io.import_window(args.files[0])
                plt.offscreen = render_offscreen
                _show_and_finalize(
                    plt,
                    args,
                    mode=interactor_mode,
                    close_on_exit=True,
                )
                return
            except KeyError:  # list of objects, create Assembly
                objs = vedo.Assembly(args.files[0])
                for i, ob in enumerate(objs):
                    if ob:
                        ob.c(i)
                plt = vedo.Plotter(offscreen=render_offscreen)
                _show_and_finalize(
                    plt,
                    args,
                    objs,
                    mode=interactor_mode,
                    close_on_exit=True,
                )
                return
        #########################################################

        if multirenderer_mode:
            plt = vedo.Plotter(
                size=wsize,
                N=N,
                bg=args.background,
                bg2=args.background_grad,
                sharecam=sharecam,
                offscreen=render_offscreen,
            )
            settings.immediate_rendering = False
            plt.axes = args.axes_type
            for i in range(N):
                plt.add_hover_legend(at=i)
            if args.axes_type in (4, 5):
                plt.axes = 0
        else:
            plt = vedo.Plotter(
                size=wsize,
                bg=args.background,
                bg2=args.background_grad,
                offscreen=render_offscreen,
            )
            plt.axes = args.axes_type
            plt.add_hover_legend()

        ds = 0
        objs = []

        for i in range(N):
            f = args.files[i]

            colb = args.color
            if args.color is None and N > 1:
                colb = i

            obj = vedo.load(f, force=args.reload)

            if isinstance(obj, (vedo.TetMesh, vedo.UnstructuredGrid)):
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

            if multirenderer_mode:
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
                    vedo.logger.error(
                        "Please do not use wildcards within single or double quotes"
                    )

        if multirenderer_mode:
            if args.output:
                _write_cli_output(plt, args)
                plt.close()
                return
            if args.offscreen:
                plt.close()
                return
            plt.interactor.Start()
            if vedo.vtk_version == (9, 2, 2):
                plt.interactor.GetRenderWindow().SetDisplayId("_0_p_void")

        else:
            # scene is empty
            if all(a is None for a in objs):
                vedo.logger.error("Could not load file(s). Quit.")
                return
            _show_and_finalize(
                plt,
                args,
                objs,
                interactive=True,
                zoom=args.zoom,
                mode=interactor_mode,
            )
        return

    ########################################################################
    # sequence mode  -s
    else:
        # print("DEBUG simple browser mode  -s")
        axes = 1 if args.axes_type == 4 else args.axes_type

        acts = vedo.load(args.files, force=args.reload)
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

        plt = vedo.applications.Browser(acts, axes=axes, offscreen=render_offscreen)
        _show_and_finalize(plt, args, zoom=args.zoom, close_on_exit=True)
