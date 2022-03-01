from setuptools import setup

try:
    VERSIONFILE = "vedo/version.py"
    verstrline = open(VERSIONFILE, "rt").read()
    verstr = verstrline.split("=")[1].replace("\n", "").replace("'", "")
except:
    verstr = "unknown"

##############################################################
setup(
    name="vedo",
    version=verstr,
    python_requires=">=3",

    license="MIT",
    license_files=['LICENSE', 'FONT.LICENSE'],

    packages=[
               "vedo",
               "vedo.examples.basic",
               "vedo.examples.advanced",
               "vedo.examples.pyplot",
               "vedo.examples.simulations",
               "vedo.examples.volumetric",
               "vedo.examples.other",
               "vedo.examples.other.dolfin",
               "vedo.examples.other.trimesh",
    ],

    package_dir={
                  'vedo': 'vedo',
                  'vedo.examples.basic': 'examples/basic',
                  'vedo.examples.advanced': 'examples/advanced',
                  'vedo.examples.pyplot': 'examples/pyplot',
                  'vedo.examples.simulations': 'examples/simulations',
                  'vedo.examples.volumetric': 'examples/volumetric',
                  'vedo.examples.other': 'examples/other',
                  'vedo.examples.other.dolfin': 'examples/other/dolfin',
                  'vedo.examples.other.trimesh': 'examples/other/trimesh',
    },

    entry_points={
        "console_scripts": ["vedo=vedo.cli:execute_cli"],
    },

    install_requires=["vtk<9.1.0", "numpy", "Deprecated"],
    include_package_data=True,

    description="A python module for scientific analysis and visualization of 3D objects and point clouds based on VTK.",
    long_description="A python module for scientific visualization, analysis of 3D objects and point clouds based on VTK. Check out https://vedo.embl.es for documentation.",

    author="Marco Musy",
    author_email="marco.musy@embl.es",
    maintainer="Marco Musy",
    url="https://github.com/marcomusy/vedo",
    keywords="vtk 3D science analysis visualization mesh numpy",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Operating System :: MacOS",
    ],
)
