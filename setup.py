from setuptools import setup

with open("vedo/version.py", "r") as fh:
    verstrline = fh.read()
    verstr = verstrline.split("=")[1].replace("'", "").strip()

##############################################################
setup(
    name="vedo",
    version=verstr,
    python_requires=">=3",

    license="MIT",
    license_files=['LICENSE', 'FONT.LICENSE'],

    description="A python module for scientific analysis and visualization of 3D objects and point clouds based on VTK and Numpy.",
    long_description="A python module for scientific visualization, analysis of 3D objects and point clouds based on VTK and Numpy. Check out https://vedo.embl.es for documentation.",

    author="Marco Musy",
    author_email="marco.musy@embl.es",
    maintainer="Marco Musy",
    url="https://github.com/marcomusy/vedo",

    keywords="vtk numpy 3D science analysis visualization mesh",
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

    packages=[
               "vedo",
               "vedo.examples.basic",
               "vedo.examples.advanced",
               "vedo.examples.pyplot",
               "vedo.examples.simulations",
               "vedo.examples.volumetric",
               "vedo.examples.other",
    ],

    package_dir={
                  'vedo': 'vedo',
                  'vedo.examples.basic': 'examples/basic',
                  'vedo.examples.advanced': 'examples/advanced',
                  'vedo.examples.pyplot': 'examples/pyplot',
                  'vedo.examples.simulations': 'examples/simulations',
                  'vedo.examples.volumetric': 'examples/volumetric',
                  'vedo.examples.other': 'examples/other',
    },

    entry_points={
        "console_scripts": ["vedo=vedo.cli:execute_cli"],
    },

    install_requires=["vtk", "numpy", "Pygments"],
    include_package_data=True,

)
