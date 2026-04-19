#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import subprocess
import time

__docformat__ = "google"


###############################################################################
def init_colab(enable_k3d=True) -> None:
    """
    Initialize a Google Colab environment for vedo.

    Installs Xvfb and pyvirtualdisplay, starts a virtual display,
    and optionally installs and enables k3d.
    """
    import vedo
    print("setting up colab environment (can take a minute) ...", end="")

    if subprocess.run(["which", "Xvfb"], capture_output=True).returncode != 0:
        os.system("apt-get install xvfb")

    os.system("pip install pyvirtualdisplay")

    from pyvirtualdisplay import Display  # type: ignore
    display = Display(visible=0)
    display.start()

    if enable_k3d:
        os.system("pip install k3d")

    from google.colab import output  # type: ignore
    output.enable_custom_widget_manager()

    if enable_k3d:
        import k3d
        try:
            print("installing k3d...", end="")
            os.system("jupyter nbextension install --py --user k3d")
            os.system("jupyter nbextension enable  --py --user k3d")
            k3d.switch_to_text_protocol()
            vedo.settings.default_backend = "k3d"
            vedo.settings.backend_autoclose = False
        except Exception:
            print("(FAILED) ... ", end="")

    print(" setup completed.")
    return display  # caller may need to keep the display alive


###############################################################################
def start_xvfb() -> None:
    """
    Start an Xvfb virtual framebuffer.

    Xvfb performs all graphical operations in virtual memory without
    showing any screen output. Useful for headless rendering on Linux.
    """
    print("starting xvfb (can take a minute) ...", end="")
    if subprocess.run(["which", "Xvfb"], capture_output=True).returncode != 0:
        subprocess.run(["apt-get", "install", "-y", "xvfb"], check=False)
    os.environ["DISPLAY"] = ":99.0"
    subprocess.Popen(
        ["Xvfb", ":99", "-screen", "0", "1024x768x24"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(3)
    print(" xvfb started.")


###############################################################################
class Settings:
    """
    Global settings for vedo. Access via the singleton ``vedo.settings``.

    Attributes can be set as properties or with dictionary syntax:

    Examples:
        ```python
        from vedo import settings, Cube
        settings.use_parallel_projection = True
        settings["use_parallel_projection"] = True  # equivalent
        Cube().color('g').show().close()
        ```

    Call ``print(vedo.settings)`` to list all current values grouped by category.

    Font parameters can be customised per font:
        ```python
        settings.font_parameters["Normografo"] = dict(
            mono=False, fscale=0.75, hspacing=1, lspacing=0.2,
            dotsep="~×", islocal=True,
        )
        # mono    : if True, all letters occupy the same horizontal slot
        # fscale  : overall scaling factor for the font size
        # hspacing: horizontal stretching (letters and words)
        # lspacing: horizontal spacing between letters (not words)
        # dotsep  : characters interpreted as dot separator
        # islocal : True = bundled in /fonts, False = downloaded from vedo.embl.es/fonts
        ```

    To run a fonts demo: ``vedo --run fonts``

    Available fonts: http://vedo.embl.es/fonts
    """

    # Restrict attributes so accidental typos raise AttributeError.
    __slots__ = [
        "default_font",
        "default_backend",
        "cache_directory",
        "palette",
        "remember_last_figure_format",
        "enable_print_color",
        "progressbar_delay",
        "enable_pipeline",
        "enable_default_mouse_callbacks",
        "enable_default_keyboard_callbacks",
        "screenshot_transparent_background",
        "screenshot_large_image",
        "force_single_precision_points",
        "enable_rendering_points_as_spheres",
        "immediate_rendering",
        "interpolate_scalars_before_mapping",
        "use_parallel_projection",
        "tiff_orientation_type",
        "point_smoothing",
        "line_smoothing",
        "polygon_smoothing",
        "light_follows_camera",
        "two_sided_lighting",
        "use_depth_peeling",
        "alpha_bit_planes",
        "multi_samples",
        "max_number_of_peels",
        "occlusion_ratio",
        "use_fxaa",
        "preserve_depth_buffer",
        "use_polygon_offset",
        "polygon_offset_factor",
        "polygon_offset_units",
        "renderer_frame_color",
        "renderer_frame_alpha",
        "renderer_frame_width",
        "renderer_frame_padding",
        "window_splitting_position",
        "background_gradient_orientation",
        "annotated_cube_color",
        "annotated_cube_text_color",
        "annotated_cube_text_scale",
        "annotated_cube_texts",
        "annotated_cube_text_rotations",
        "backend_autoclose",
        "k3d_menu_visibility",
        "k3d_plot_height",
        "k3d_antialias",
        "k3d_lighting",
        "k3d_camera_autofit",
        "k3d_grid_visible",
        "k3d_grid_autofit",
        "k3d_axes_color",
        "k3d_axes_helper",
        "k3d_point_shader",
        "k3d_line_shader",
        "font_parameters",
    ]

    # Groups used by __str__ to produce readable output.
    # Each entry is (section_title, [slot_names]).
    _SECTIONS = (
        ("General", [
            "default_font",
            "default_backend",
            "cache_directory",
            "palette",
            "remember_last_figure_format",
            "enable_print_color",
            "progressbar_delay",
        ]),
        ("Interaction", [
            "enable_pipeline",
            "enable_default_mouse_callbacks",
            "enable_default_keyboard_callbacks",
        ]),
        ("Screenshots", [
            "screenshot_transparent_background",
            "screenshot_large_image",
        ]),
        ("Rendering", [
            "force_single_precision_points",
            "enable_rendering_points_as_spheres",
            "immediate_rendering",
            "interpolate_scalars_before_mapping",
            "use_parallel_projection",
            "tiff_orientation_type",
        ]),
        ("Smoothing", [
            "point_smoothing",
            "line_smoothing",
            "polygon_smoothing",
        ]),
        ("Lighting", [
            "light_follows_camera",
            "two_sided_lighting",
        ]),
        ("Transparency / Anti-aliasing", [
            "use_depth_peeling",
            "alpha_bit_planes",
            "multi_samples",
            "max_number_of_peels",
            "occlusion_ratio",
            "use_fxaa",
            "preserve_depth_buffer",
        ]),
        ("Polygon offset", [
            "use_polygon_offset",
            "polygon_offset_factor",
            "polygon_offset_units",
        ]),
        ("Window / Multi-renderer", [
            "renderer_frame_color",
            "renderer_frame_alpha",
            "renderer_frame_width",
            "renderer_frame_padding",
            "window_splitting_position",
            "background_gradient_orientation",
        ]),
        ("Annotated cube", [
            "annotated_cube_color",
            "annotated_cube_text_color",
            "annotated_cube_text_scale",
            "annotated_cube_texts",
            "annotated_cube_text_rotations",
        ]),
        ("Backend", [
            "backend_autoclose",
        ]),
        ("K3D backend", [
            "k3d_menu_visibility",
            "k3d_plot_height",
            "k3d_antialias",
            "k3d_lighting",
            "k3d_camera_autofit",
            "k3d_grid_visible",
            "k3d_grid_autofit",
            "k3d_axes_color",
            "k3d_axes_helper",
            "k3d_point_shader",
            "k3d_line_shader",
        ]),
        ("Fonts", [
            "font_parameters",
        ]),
    )

    # Dry-run mode (for testing only):
    #   0 = normal execution
    #   1 = do not hold execution (no interactive loop)
    #   2 = do not hold execution and do not open any window
    # Stored as a class variable (not an instance slot) so that vtkclasses.py
    # can read Settings.dry_run_mode without instantiating Settings and without
    # triggering circular imports.
    dry_run_mode = 0

    ############################################################
    def __init__(self) -> None:

        self.default_backend = "vtk"
        try:
            # adapted from: https://stackoverflow.com/a/39662359/2912349
            shell = get_ipython().__class__.__name__  # type: ignore
            if shell == "ZMQInteractiveShell":
                self.default_backend = "2d"
        except NameError:
            pass

        self.default_font = "Normografo"
        self.palette = 0
        self.remember_last_figure_format = False
        self.enable_print_color = True
        self.progressbar_delay = 0.5

        self.enable_pipeline = True
        self.enable_default_mouse_callbacks = True
        self.enable_default_keyboard_callbacks = True

        if "VEDO_CACHE_DIR" in os.environ:
            self.cache_directory = os.environ["VEDO_CACHE_DIR"]
        else:
            self.cache_directory = ".cache"  # "/vedo" is appended automatically

        self.screenshot_transparent_background = False
        self.screenshot_large_image = False

        self.force_single_precision_points = True
        self.enable_rendering_points_as_spheres = True
        self.immediate_rendering = True
        self.interpolate_scalars_before_mapping = True
        self.use_parallel_projection = False
        self.tiff_orientation_type = 1

        self.point_smoothing = False
        self.line_smoothing = False
        self.polygon_smoothing = False

        self.light_follows_camera = False
        self.two_sided_lighting = True

        self.use_depth_peeling = False
        self.alpha_bit_planes = True
        self.multi_samples = 8
        self.max_number_of_peels = 4
        self.occlusion_ratio = 0.1
        self.use_fxaa = False
        self.preserve_depth_buffer = False

        self.use_polygon_offset = True
        self.polygon_offset_factor = 0.1
        self.polygon_offset_units = 0.1

        self.renderer_frame_color = None
        self.renderer_frame_alpha = 0.5
        self.renderer_frame_width = 0.5
        self.renderer_frame_padding = 0.0001
        self.window_splitting_position = None
        self.background_gradient_orientation = 0

        self.annotated_cube_color = (0.75, 0.75, 0.75)
        self.annotated_cube_text_color = None
        self.annotated_cube_text_scale = 0.2
        self.annotated_cube_texts = [
            "right",
            "left ",
            "front",
            "back ",
            " top ",
            "bottom",
        ]
        self.annotated_cube_text_rotations = [0, 0, 90]

        self.backend_autoclose = True

        self.k3d_menu_visibility = True
        self.k3d_plot_height = 512
        self.k3d_antialias = True
        self.k3d_lighting = 1.5
        self.k3d_camera_autofit = True
        self.k3d_grid_visible = None
        self.k3d_grid_autofit = True
        self.k3d_axes_color = "k4"
        self.k3d_axes_helper = 1.0
        self.k3d_point_shader = "mesh"
        self.k3d_line_shader = "thick"

        self.font_parameters = dict(
            Normografo=dict(
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~×",
                islocal=True,
            ),
            Bongas=dict(
                mono=False,
                fscale=0.875,
                hspacing=0.52,
                lspacing=0.25,
                dotsep="·",
                islocal=True,
            ),
            Calco=dict(
                mono=True,
                fscale=0.8,
                hspacing=1,
                lspacing=0.1,
                dotsep="×",
                islocal=True,
            ),
            Comae=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~×",
                islocal=True,
            ),
            ComicMono=dict(
                mono=True,
                fscale=0.8,
                hspacing=1,
                lspacing=0.1,
                dotsep="x",
                islocal=False,
            ),
            Edo=dict(
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~x ",
                islocal=False,
            ),
            FiraMonoMedium=dict(
                mono=True,
                fscale=0.8,
                hspacing=1,
                lspacing=0.1,
                dotsep="×",
                islocal=False,
            ),
            FiraMonoBold=dict(
                mono=True,
                fscale=0.8,
                hspacing=1,
                lspacing=0.1,
                dotsep="×",
                islocal=False,
            ),
            Glasgo=dict(
                mono=True,
                fscale=0.75,
                lspacing=0.1,
                hspacing=1,
                dotsep="~×",
                islocal=True,
            ),
            Kanopus=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.15,
                hspacing=0.75,
                dotsep="~×",
                islocal=True,
            ),
            LogoType=dict(
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~×",
                islocal=False,
            ),
            Quikhand=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.6,
                lspacing=0.15,
                dotsep="~~×~",
                islocal=True,
            ),
            SmartCouric=dict(
                mono=True,
                fscale=0.8,
                hspacing=1.05,
                lspacing=0.1,
                dotsep="×",
                islocal=True,
            ),
            Spears=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.5,
                lspacing=0.2,
                dotsep="~×",
                islocal=False,
            ),
            Theemim=dict(
                mono=False,
                fscale=0.825,
                hspacing=0.52,
                lspacing=0.3,
                dotsep="~~×",
                islocal=True,
            ),
            VictorMono=dict(
                mono=True,
                fscale=0.725,
                hspacing=1,
                lspacing=0.1,
                dotsep="×",
                islocal=True,
            ),
            Justino1=dict(
                mono=True,
                fscale=0.725,
                hspacing=1,
                lspacing=0.1,
                dotsep="×",
                islocal=False,
            ),
            Justino2=dict(
                mono=True,
                fscale=0.725,
                hspacing=1,
                lspacing=0.1,
                dotsep="×",
                islocal=False,
            ),
            Justino3=dict(
                mono=True,
                fscale=0.725,
                hspacing=1,
                lspacing=0.1,
                dotsep="×",
                islocal=False,
            ),
            Calibri=dict(
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~×",
                islocal=False,
            ),
            Capsmall=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.75,
                lspacing=0.15,
                dotsep="~×",
                islocal=False,
            ),
            Cartoons123=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.75,
                lspacing=0.15,
                dotsep="x",
                islocal=False,
            ),
            Darwin=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.75,
                lspacing=0.15,
                dotsep="x",
                islocal=False,
            ),
            Vega=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.75,
                lspacing=0.15,
                dotsep="×",
                islocal=False,
            ),
            Meson=dict(
                mono=False,
                fscale=0.8,
                hspacing=0.9,
                lspacing=0.225,
                dotsep="~×",
                islocal=False,
            ),
            Komika=dict(
                mono=False,
                fscale=0.7,
                hspacing=0.75,
                lspacing=0.225,
                dotsep="~×",
                islocal=False,
            ),
            Brachium=dict(
                mono=True,
                fscale=0.8,
                hspacing=1,
                lspacing=0.1,
                dotsep="x",
                islocal=False,
            ),
            Dalim=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~×",
                islocal=False,
            ),
            Miro=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~×",
                islocal=False,
            ),
            Ubuntu=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~×",
                islocal=False,
            ),
            Mizar=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=0.75,
                dotsep="~×",
                islocal=False,
            ),
            LiberationSans=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~×",
                islocal=False,
            ),
            DejavuSansMono=dict(
                mono=True,
                fscale=0.725,
                hspacing=1,
                lspacing=0.1,
                dotsep="~×",
                islocal=False,
            ),
            SunflowerHighway=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~×",
                islocal=False,
            ),
            Swansea=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=1,
                dotsep="~×",
                islocal=False,
            ),
            Housekeeper=dict(  # supports chinese glyphs
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~×",
                islocal=False,
            ),
            Wananti=dict(  # supports chinese glyphs
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~x",
                islocal=False,
            ),
            AnimeAce=dict(
                mono=False,
                fscale=0.75,
                hspacing=1,
                lspacing=0.2,
                dotsep="~x",
                islocal=False,
            ),
            Antares=dict(
                mono=True,
                fscale=0.8,
                hspacing=1,
                lspacing=0.1,
                dotsep="×",
                islocal=False,
            ),
            Archistico=dict(
                mono=False,
                fscale=0.75,
                lspacing=0.2,
                hspacing=0.75,
                dotsep="~×",
                islocal=False,
            ),
            KazyCase=dict(
                mono=True,
                fscale=0.8,
                hspacing=1.2,
                lspacing=0.1,
                dotsep="×",
                islocal=False,
            ),
            Roboto=dict(
                mono=True,
                fscale=0.8,
                hspacing=1,
                lspacing=0.1,
                dotsep="×",
                islocal=False,
            ),
        )

    ############################################################
    def __getitem__(self, key):
        """Make the class work like a dictionary too."""
        try:
            return getattr(self, key)
        except AttributeError:
            raise KeyError(key) from None

    def __setitem__(self, key, value):
        """Make the class work like a dictionary too."""
        setattr(self, key, value)

    def __contains__(self, key):
        """Support ``'key' in settings`` queries."""
        return key in self.__slots__ or key == "dry_run_mode"

    def __setattr__(self, key, value):
        # dry_run_mode is a class variable (not a slot) so that vtkclasses.py
        # can read Settings.dry_run_mode without needing an instance.
        # Redirect instance assignment to the class to keep the two in sync.
        if key == "dry_run_mode":
            type(self).dry_run_mode = value
            return
        super().__setattr__(key, value)

    def _body_lines(self) -> list:
        """Return plain-text lines for all settings, grouped by section."""
        lines = []
        for title, keys in self._SECTIONS:
            lines.append(f"\n# {title}")
            for key in keys:
                val = getattr(self, key)
                if key == "font_parameters":
                    lines.append(f"font_parameters  # {len(val)} fonts defined")
                else:
                    lines.append(f"{key} = {val!r}")
        lines.append("\n# Testing")
        lines.append(f"dry_run_mode = {type(self).dry_run_mode!r}")
        return lines

    def __str__(self) -> str:
        s = "\n".join(self._body_lines())
        try:
            from pygments import highlight
            from pygments.lexers import Python3Lexer
            from pygments.formatters import Terminal256Formatter
            s = highlight(s, Python3Lexer(), Terminal256Formatter(style="zenburn"))
        except (ModuleNotFoundError, ImportError):
            pass
        module = self.__class__.__module__
        name = self.__class__.__name__
        header = f"{module}.{name}".ljust(75)
        return f"\x1b[1m\x1b[7m{header}\x1b[0m\n" + s.strip()

    def __repr__(self) -> str:
        return self.__str__()

    def __rich__(self):
        from rich.panel import Panel
        from rich.syntax import Syntax

        body = "\n".join(self._body_lines()).strip()
        module = self.__class__.__module__
        name = self.__class__.__name__
        return Panel(
            Syntax(body, "python", theme="zenburn"),
            title=f"{module}.{name}",
            title_align="left",
            border_style="bold white",
        )

    ############################################################
    def keys(self) -> list:
        """Return all setting names."""
        return list(self.__slots__) + ["dry_run_mode"]

    def values(self) -> list:
        """Return all current setting values."""
        return [getattr(self, key) for key in self.__slots__] + [type(self).dry_run_mode]

    def items(self) -> list:
        """Return all (name, value) pairs."""
        return [(key, getattr(self, key)) for key in self.__slots__] + [("dry_run_mode", type(self).dry_run_mode)]

    def reset(self) -> None:
        """Reset all settings to their default values."""
        self.__init__()

    ############################################################
    def clear_cache(self) -> None:
        """Clear the vedo cache directory."""
        import shutil
        cachedir = self.cache_directory
        if not os.path.isabs(cachedir):
            cachedir = os.path.join(os.path.expanduser("~"), cachedir)
        cachedir = os.path.join(cachedir, "vedo")
        try:
            shutil.rmtree(cachedir)
            print(f"Cache directory '{cachedir}' cleared.")
        except FileNotFoundError:
            print(f"Cache directory '{cachedir}' not found.")

    ############################################################
    def set_vtk_verbosity(self, level: int) -> None:
        """
        Set the verbosity level of VTK messages on stderr.

        Args:
            level: 0 = errors only, 1 = warnings, 2 = info
        """
        from vedo.vtkclasses import vtk
        vtkLogger = vtk.vtkLogger
        levels = {
            0: vtkLogger.VERBOSITY_ERROR,
            1: vtkLogger.VERBOSITY_WARNING,
            2: vtkLogger.VERBOSITY_INFO,
        }
        if level not in levels:
            raise ValueError(f"Invalid VTK verbosity level {level!r}. Allowed values: 0, 1, 2.")
        vtkLogger.SetStderrVerbosity(levels[level])
