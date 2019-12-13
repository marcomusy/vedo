from setuptools import setup

try:
    VERSIONFILE = "vtkplotter/version.py"
    verstrline = open(VERSIONFILE, "rt").read()
    verstr = verstrline.split('=')[1].replace('\n','').replace("'","")
except:
    verstr='unknown'

##############################################################
setup(
    name='vtkplotter',
    version=verstr,
    packages=['vtkplotter'],
    scripts=['bin/vtkplotter', 'bin/vtkconvert'],
    install_requires=['vtk', 'numpy'],
    description='''A python module for scientific visualization,
    analysis and animation of 3D objects and point clouds based on VTK.''',
    long_description="""A python module for scientific visualization,
    analysis and animation of 3D objects and point clouds based on VTK.
    Check out https://vtkplotter.embl.es for documentation.""",
    author='Marco Musy',
    author_email='marco.musy@gmail.com',
    license='MIT',
    url='https://github.com/marcomusy/vtkplotter',
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
                'Programming Language :: Python :: 3.7'
                ],
    include_package_data=True
)


##############################################################
# # check examples
# change version in vtkplotter/version.py

# cd ~/Projects/vtkplotter/
# pip install .

# cd examples && ./run_all.sh
# cd ~/Projects/vtkplotter/
# python prove/test_filetypes.py

# check vtkconvert:
# vtkconvert vtkplotter/data/290.vtk -to ply; vtkplotter vtkplotter/data/290.ply

# check on python2 the same stuff is ok
# cd ~/Projects/vtkplotter/
# sudo -H pip install . -U
# python examples/tutorial.py

# check notebooks:
# cd ~/Projects/vtkplotter/
# jupyter notebook > /dev/null 2>&1
# remove trailing spaces

# cd ~/Projects/vtkplotter/
# rm -rf examples/*/.ipynb_checkpoints examples/*/*/.ipynb_checkpoints .ipynb_checkpoints/
# rm -rf examples/other/dolfin/navier_stokes_cylinder/ examples/other/dolfin/shuttle.xml
# rm examples/other/trimesh/featuretype.STL examples/other/trimesh/machinist.XAML
# rm examples/other/scene.npy examples/other/timecourse1d.npy vtkplotter/data/290.ply
# rm examples/other/voronoi3d.txt examples/other/voronoi3d.txt.vol
# rm examples/other/embryo.html examples/other/embryo.x3d

# git status
# git add [files]
# git commit -a -m 'comment'
# git push

# git status
# (sudo apt install twine)
# (python -m pip install --user --upgrade twine)
# python setup.py sdist bdist_wheel
# twine upload dist/vtkplotter-?.?.?.tar.gz -r pypi
# make release

## to generate documentation:
# Install the dependencies in docs/requirements.txt
#  pip install -r docs/requirements.txt
#
# Run the documentaion generation:
#  cd docs
#  make html
# Open the HTML webpage
#  open build/html/index.html
# check if dolfin shows up
#
# mount_staging
# cp -r build/html/* ~/Projects/StagingServer/var/www/html/vtkplotter.embl.es/
# version bump vtkplotter/version.py


## to generate gif: ezgif.com

