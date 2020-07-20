from setuptools import setup

try:
    VERSIONFILE = "vedo/version.py"
    verstrline = open(VERSIONFILE, "rt").read()
    verstr = verstrline.split('=')[1].replace('\n','').replace("'","")
except:
    verstr='unknown'

##############################################################
setup(
    name='vedo',
    version=verstr,
    packages=['vedo'],
    scripts=['bin/vedo', 'bin/vedo-convert'],
    install_requires=['vtk', 'numpy'],
    description='A python module for scientific analysis and visualization of 3D objects and point clouds based on VTK.',
    long_description="""A python module for scientific visualization,
analysis of 3D objects and point clouds based on VTK.
Check out https://vedo.embl.es for documentation.""",
    author='Marco Musy',
    author_email='marco.musy@gmail.com',
    license='MIT',
    url='https://github.com/marcomusy/vedo',
    keywords='vtk 3D visualization mesh numpy',
    classifiers=['Intended Audience :: Science/Research',
                'Intended Audience :: Education',
                'Intended Audience :: Information Technology',
                'Programming Language :: Python',
                'License :: OSI Approved :: MIT License',
                'Topic :: Scientific/Engineering :: Visualization',
                'Topic :: Scientific/Engineering :: Physics',
                'Topic :: Scientific/Engineering :: Medical Science Apps.',
                'Topic :: Scientific/Engineering :: Information Analysis',
                'Programming Language :: Python :: 2.7',
                'Programming Language :: Python :: 3.5',
                'Programming Language :: Python :: 3.6',
                'Programming Language :: Python :: 3.7',
                'Programming Language :: Python :: 3.8',
                ],
    include_package_data=True
)


























