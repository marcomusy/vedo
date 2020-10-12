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
    packages=[
               "vedo",
               "vedo.examples",
               "vedo.examples.basic",
               "vedo.examples.advanced",
               "vedo.examples.pyplot",
               "vedo.examples.simulations",
               "vedo.examples.tetmesh",
               "vedo.examples.volumetric",
               "vedo.examples.other.dolfin",
               "vedo.examples.other.trimesh",
    ],
    package_dir={
                  'vedo': 'vedo',
                  'vedo.examples': 'examples',
                  'vedo.examples.basic': 'examples/basic',
                  'vedo.examples.advanced': 'examples/advanced',
                  'vedo.examples.pyplot': 'examples/pyplot',
                  'vedo.examples.simulations': 'examples/simulations',
                  'vedo.examples.tetmesh': 'examples/tetmesh',
                  'vedo.examples.volumetric': 'examples/volumetric',
                  'vedo.examples.other.dolfin': 'examples/other/dolfin',
                  'vedo.examples.other.trimesh': 'examples/other/trimesh',
    },
    scripts=["bin/vedo",
             "bin/vedo-convert",
             ],
    #entry_points={
    #    'console_scripts': [
    #        "vedo = bin/vedo:main"
    #    ]
    #},
    install_requires=["vtk<9.0.0", "numpy"],
    description="A python module for scientific analysis and visualization of 3D objects and point clouds based on VTK.",
    long_description="A python module for scientific visualization, analysis of 3D objects and point clouds based on VTK. Check out https://vedo.embl.es for documentation.",
    author="Marco Musy",
    author_email="marco.musy@embl.es",
    license="MIT",
    url="https://github.com/marcomusy/vedo",
    keywords="vtk 3D visualization mesh numpy",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Intended Audience :: Education",
        "Intended Audience :: Information Technology",
        "Programming Language :: Python",
        "License :: OSI Approved :: MIT License",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Programming Language :: Python :: 2.7",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
    ],
    include_package_data=True,
)
