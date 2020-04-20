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
    scripts=['bin/vtkplotter', 'bin/vtkplotter-convert'],
    install_requires=['vtk'],
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
                'Programming Language :: Python :: 3.7',
                'Programming Language :: Python :: 3.8',
                ],
    include_package_data=True
)





##############################################################
# # check examples
# change version in vtkplotter/version.py

# cd ~/Projects/vtkplotter/
# remove trailing spaces
# pip install .

# cd ~/Projects/vtkplotter/examples/
# ./run_all.sh
# cd ~/Projects/vtkplotter/
# python prove/test_filetypes.py

# cd ~/Projects/vtkplotter/tests/common
# ./run_all.sh
# check also scripts in docs

# check vtkconvert:
# vtkplotter-convert data/290.vtk -to ply; vtkplotter data/290.ply

# check on python2 the same stuff is ok
# cd ~/Projects/vtkplotter/
# sudo -H pip install . -U
# python ~/Projects/vtkplotter/examples/tutorial.py

# check notebooks:
# cd ~/Projects/vtkplotter/notebooks/
# jupyter notebook > /dev/null 2>&1

# cd ~/Projects/vtkplotter/
# rm -rf examples/*/.ipynb_checkpoints examples/*/*/.ipynb_checkpoints .ipynb_checkpoints/
# rm -rf examples/other/dolfin/navier_stokes_cylinder/ examples/other/dolfin/shuttle.xml
# rm examples/other/trimesh/featuretype.STL examples/other/trimesh/machinist.XAML
# rm examples/other/scene.npy examples/other/timecourse1d.npy vtkplotter/data/290.ply
# rm examples/other/voronoi3d.txt examples/other/voronoi3d.txt.vol
# rm examples/other/embryo.html examples/other/embryo.x3d

# comment out defs in docs.py

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
# release examples

## to generate documentation:
# comment in defs in docs.py
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


######################## fenics 2019.2 docker:

# To copy files over:
# sudo docker ps   # to know the container name
# cd Projects
# sudo docker cp  vtkplotter admiring_panini:/home/fenics/shared/
# sudo chmod -R 755 vtkplotter*/
#
# cd vtkplotter
# sudo pip install .
# sudo apt update -y
# sudo apt upgrade -y
# sudo apt install python3-vtk7 -y
# docker pull quay.io/fenicsproject/stable:latest
# docker pull quay.io/fenicsproject/dolfinx:dev-env-real
#
# docker run -ti -v $(pwd):/home/musy/my-project/shared --name fenics-container quay.io/fenicsproject/dolfinx:dev-env-real
#
#    cd
#    pip3 install vtkplotter # OR
#    git clone https://github.com/marcomusy/vtkplotter.git
#    cd vtkplotter
#    pip3 -v install . --user
#
#    cd
#    pip3 install git+https://github.com/FEniCS/fiat.git --upgrade
#    pip3 install git+https://github.com/FEniCS/ufl.git  --upgrade
#    pip3 install git+https://github.com/FEniCS/ffcx.git --upgrade
#    git clone https://github.com/FEniCS/dolfinx.git
#    cd dolfinx
#    mkdir -p build && cd build && cmake -G Ninja -DCMAKE_BUILD_TYPE=Developer ../cpp/
#    ninja -j3 install
#    cd ../python
#    pip3 -v install . --user


























