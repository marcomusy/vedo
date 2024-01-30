#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

__docformat__ = "google"


class Settings:
    """
    General settings to modify the global behavior and style.

    Example:
        ```python
        from vedo import settings, Cube
        settings.use_parallel_projection = True
        # settings["use_parallel_projection"] = True  # this is equivalent!
        Cube().color('g').show().close()
        ```

    List of available properties:

    ```python
    # Set the default font to be used for axes, comments etc.
    # Check out the available fonts at http://vedo.embl.es/fonts
    # For example:
    default_font = 'Normografo'
    # To customize the font parameters use:
    settings.font_parameters["Normografo"] = dict(
        mono=False,
        fscale=0.75,
        hspacing=1,
        lspacing=0.2,
        dotsep="~×",
        islocal=True,
    )
    # Where
    # mono    : if True all letters occupy the same space slot horizontally
    # fscale  : sets the general scaling factor for the size of the font
    # hspacing: horizontal stretching factor (affects both letters and words)
    # lspacing: horizontal spacing inbetween letters (not words)
    # dotsep  : a string of characters to be interpreted as dot separator
    # islocal : if locally stored in /fonts, otherwise it's on vedo.embl.es/fonts
    #
    # To run a demo try:
    # vedo --run fonts

    # Use this local folder to store downloaded files (default is ~/.cache/vedo)
    cache_directory = ".cache"

    # Palette number when using an integer to choose a color
    palette = 0

    # Options for saving window screenshots:
    screenshot_transparent_background = False
    screeshot_large_image = False # sometimes setting this to True gives better results

    # Enable tracking pipeline functionality:
    # allows to show a graph with the pipeline of action which let to a final object
    # this is achieved by calling "myobj.pipeline.show()" (a new window will pop up)
    self.enable_pipeline = True

    # Remember the last format used when creating new figures in vedo.pyplot
    # this is useful when creating multiple figures of the same kind
    # and avoid to specify the format each time in plot(..., like=...)
    remember_last_figure_format = False

    # Set up default mouse and keyboard callbacks
    enable_default_mouse_callbacks = True
    enable_default_keyboard_callbacks = True

    # Progress bar delay before showing up [sec]
    self.progressbar_delay = 0.5

    # If False, when multiple renderers are present, render only once at the end
    immediate_rendering = True

    # In multirendering mode, show a grey frame margin (set width=0 to disable)
    renderer_frame_color = None
    renderer_frame_alpha = 0.5
    renderer_frame_width = 0.5
    renderer_frame_padding = 0.0001

    # In multirendering mode, set the position of the horizontal of vertical splitting [0,1]
    window_splitting_position = None

    # Gradient orientation mode for background window color
    # 0 = Vertical
    # 1 = Horizontal
    # 2 = Radial viewport farthest side
    # 3 = Radial viewport farthest corner
    background_gradient_orientation = 0

    # Enable / disable color printing by printc()
    enable_print_color = True

    # Smoothing options for points, lines and polygons
    point_smoothing = False
    line_smoothing = False
    polygon_smoothing = False

    # Turn on/off the automatic repositioning of lights as the camera moves
    light_follows_camera = False
    two_sided_lighting = True

    # Turn on/off rendering of translucent material with depth peeling technique
    use_depth_peeling = False
    alpha_bit_planes  = True   # options only active if useDepthPeeling=True
    multi_samples     = 8      # antialiasing multisample buffer
    max_number_of_peels= 4     # maximum number of rendering passes
    occlusion_ratio   = 0.0    # occlusion ratio, 0 = exact image.

    # Turn on/off nvidia FXAA post-process anti-aliasing, if supported
    use_fxaa = False           # either True or False

    # By default, the depth buffer is reset for each renderer
    #  If True, use the existing depth buffer
    preserve_depth_buffer = False

    # Use a polygon/edges offset to possibly resolve conflicts in rendering
    use_polygon_offset    = False
    polygon_offset_factor = 0.1
    polygon_offset_units  = 0.1

    # Interpolate scalars to render them smoothly
    interpolate_scalars_before_mapping = True

    # Set parallel projection On or Off (place camera to infinity, no perspective effects)
    use_parallel_projection = False

    # Set orientation type when reading TIFF files:
    # TOPLEFT  1 (row 0 top,    col 0 lhs)    TOPRIGHT 2 (row 0 top,    col 0 rhs)
    # BOTRIGHT 3 (row 0 bottom, col 0 rhs)    BOTLEFT  4 (row 0 bottom, col 0 lhs)
    # LEFTTOP  5 (row 0 lhs,    col 0 top)    RIGHTTOP 6 (row 0 rhs,    col 0 top)
    # RIGHTBOT 7 (row 0 rhs,    col 0 bottom) LEFTBOT  8 (row 0 lhs,    col 0 bottom)
    tiff_orientation_type = 1

    # Annotated cube axis type nr. 5 options:
    annotated_cube_color      = (0.75, 0.75, 0.75)
    annotated_cube_text_color = None # use default, otherwise specify a single color
    annotated_cube_text_scale = 0.2
    annotated_cube_texts      = ["right","left ", "front","back ", " top ", "bttom"]

    # Set the default backend for plotting in jupyter notebooks.
    # If a jupyter environment is detected, the default is automatically switched to "2d"
    default_backend = "vtk"

    # Automatically close the Plotter instance after show() in jupyter sessions
    # setting it to False will keep the current Plotter instance active
    backend_autoclose = True

    # Settings specific to the K3D backend in jupyter notebooks
    k3d_menu_visibility = True
    k3d_plot_height   = 512
    k3d_antialias     = True
    k3d_lighting      = 1.5
    k3d_camera_autofit= True
    k3d_grid_autofit  = True
    k3d_axes_color    = "gray4"
    k3d_axes_helper   = 1.0     # size of the small triad of axes on the bottom right
    k3d_point_shader  = "mesh"  # others are '3d', '3dSpecular', 'dot', 'flat'
    k3d_line_shader   = "thick" # others are 'flat', 'mesh'
    ```
    """

    # Restrict the attributes so accidental typos will generate an AttributeError exception
    __slots__ = [
        "default_font",
        "default_backend",
        "cache_directory",
        "palette",
        "remember_last_figure_format",
        "screenshot_transparent_background",
        "screeshot_large_image",
        "enable_default_mouse_callbacks",
        "enable_default_keyboard_callbacks",
        "enable_pipeline",
        "progressbar_delay",
        "immediate_rendering",
        "renderer_frame_color",
        "renderer_frame_alpha",
        "renderer_frame_width",
        "renderer_frame_padding",
        "point_smoothing",
        "line_smoothing",
        "polygon_smoothing",
        "light_follows_camera",
        "two_sided_lighting",
        "use_depth_peeling",
        "multi_samples",
        "alpha_bit_planes",
        "max_number_of_peels",
        "occlusion_ratio",
        "use_fxaa",
        "preserve_depth_buffer",
        "use_polygon_offset",
        "polygon_offset_factor",
        "polygon_offset_units",
        "interpolate_scalars_before_mapping",
        "use_parallel_projection",
        "background_gradient_orientation",
        "window_splitting_position",
        "tiff_orientation_type",
        "annotated_cube_color",
        "annotated_cube_text_color",
        "annotated_cube_text_scale",
        "annotated_cube_texts",
        "enable_print_color",
        "backend_autoclose",
        "k3d_menu_visibility",
        "k3d_plot_height",
        "k3d_antialias",
        "k3d_lighting",
        "k3d_camera_autofit",
        "k3d_grid_autofit",
        "k3d_axes_color",
        "k3d_axes_helper",
        "k3d_point_shader",
        "k3d_line_shader",
        "font_parameters",
    ]

    ############################################################
    # Dry run mode (for test purposes only)
    # 0 = normal
    # 1 = do not hold execution
    # 2 = do not hold execution and do not show any window
    dry_run_mode = 0

    ############################################################
    def __init__(self):

        self.default_backend = "vtk"
        try:
            get_ipython()
            self.default_backend = "2d"
        except NameError:
            pass

        self.default_font = "Normografo"

        self.enable_pipeline = True
        self.progressbar_delay = 0.5
        self.palette = 0
        self.remember_last_figure_format = False

        self.cache_directory = ".cache"  # "/vedo" is added automatically

        self.screenshot_transparent_background = False
        self.screeshot_large_image = False

        self.enable_default_mouse_callbacks = True
        self.enable_default_keyboard_callbacks = True
        self.immediate_rendering = True

        self.renderer_frame_color = None
        self.renderer_frame_alpha = 0.5
        self.renderer_frame_width = 0.5
        self.renderer_frame_padding = 0.0001
        self.background_gradient_orientation = 0

        self.point_smoothing = False
        self.line_smoothing = False
        self.polygon_smoothing = False

        self.light_follows_camera = False
        self.two_sided_lighting = True

        self.use_depth_peeling = False
        self.multi_samples = 8
        self.alpha_bit_planes = 1
        self.max_number_of_peels = 4
        self.occlusion_ratio = 0.1

        self.use_fxaa = False

        self.preserve_depth_buffer = False

        self.use_polygon_offset = True
        self.polygon_offset_factor = 0.1
        self.polygon_offset_units = 0.1

        self.interpolate_scalars_before_mapping = True

        self.use_parallel_projection = False

        self.window_splitting_position = None

        self.tiff_orientation_type = 1

        self.annotated_cube_color = (0.75, 0.75, 0.75)
        self.annotated_cube_text_color = None
        self.annotated_cube_text_scale = 0.2
        self.annotated_cube_texts = ["right", "left ", "front", "back ", " top ", "bttom"]

        self.enable_print_color = True

        self.backend_autoclose = True

        self.k3d_menu_visibility = True
        self.k3d_plot_height = 512
        self.k3d_antialias   = True
        self.k3d_lighting    = 1.5
        self.k3d_camera_autofit = True
        self.k3d_grid_autofit= True
        self.k3d_axes_color  = "k4"
        self.k3d_axes_helper = 1.0
        self.k3d_point_shader= "mesh"
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

    ####################################################################################
    def __getitem__(self, key):
        """Make the class work like a dictionary too"""
        return getattr(self, key)

    def __setitem__(self, key, value):
        """Make the class work like a dictionary too"""
        setattr(self, key, value)

    def __str__(self) -> str:
        """Return a string representation of the object"""
        s = Settings.__doc__.replace("   ", "")
        s = s.replace(".. code-block:: python\n", "")
        s = s.replace("```python\n", "")
        s = s.replace("```\n", "")
        s = s.replace("\n\n", "\n #------------------------------------------------------\n")
        s = s.replace("\n  ", "\n")
        s = s.replace("\n ", "\n")
        s = s.replace(" from", "from")
        try:
            from pygments import highlight
            from pygments.lexers import Python3Lexer
            from pygments.formatters import Terminal256Formatter
            s = highlight(s, Python3Lexer(), Terminal256Formatter(style="zenburn"))
        except (ModuleNotFoundError, ImportError):
            pass

        module = self.__class__.__module__
        name = self.__class__.__name__
        header = f"{module}.{name} at ({hex(id(self))})".ljust(75)
        s = f"\x1b[1m\x1b[7m{header}\x1b[0m\n" + s
        return s.strip()

    ############################################################
    def keys(self):
        """Return all keys"""
        return self.__slots__

    def values(self):
        """Return all values"""
        return [getattr(self, key) for key in self.__slots__]

    def items(self):
        """Return all items"""
        return [(key, getattr(self, key)) for key in self.__slots__]

    def reset(self):
        """Reset all settings to their default status."""
        self.__init__()

    ############################################################
    def init_colab(self, enable_k3d=True):
        """
        Initialize colab environment
        """
        print("setting up colab environment (can take a minute) ...", end="")

        res = os.system("which Xvfb")
        if res:
            os.system("apt-get install xvfb")

        os.system("pip install pyvirtualdisplay")

        from pyvirtualdisplay import Display
        Display(visible=0).start()

        if enable_k3d:
            os.system("pip install k3d")

        from google.colab import output
        output.enable_custom_widget_manager()

        if enable_k3d:
            import k3d
            try:
                print("installing k3d...", end="")
                os.system("jupyter nbextension install --py --user k3d")
                os.system("jupyter nbextension enable  --py --user k3d")
                k3d.switch_to_text_protocol()
                self.default_backend = "k3d"
                self.backend_autoclose = False
            except:
                print("(FAILED) ... ", end="")

        print(" setup completed.")

    ############################################################
    def start_xvfb(self):
        """
        Start xvfb.

        Xvfb or X virtual framebuffer is a display server implementing
        the X11 display server protocol. In contrast to other display servers,
        Xvfb performs all graphical operations in virtual memory
        without showing any screen output.
        """
        print("starting xvfb (can take a minute) ...", end="")
        res = os.system("which Xvfb")
        if res:
            os.system("apt-get install xvfb")
        os.system("set -x")
        os.system("export DISPLAY=:99.0")
        os.system("Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &")
        os.system("sleep 3")
        os.system("set +x")
        os.system('exec "$@"')
        print(" xvfb started.")
