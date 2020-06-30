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
    install_requires=['vtk'],
    description='''A python module for scientific visualization,
    analysis and animation of 3D objects and point clouds based on VTK.''',
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





##########################################################################
# sudo -H pip install -U git+https://github.com/marcomusy/vedo.git
# change version in vedo/version.py

# cd ~/Projects/vedo/
# remove trailing spaces
# pip install .

### check examples
# cd ~/Projects/vedo/examples/
# ./run_all.sh
# cd ~/Projects/vedo/
# python prove/test_filetypes.py

# cd ~/Projects/vedo/tests/common
# ./run_all.sh &
# check also scripts in docs

# check vedo-convert:
# cd ~/Projects/vedo
# vedo-convert data/290.vtk -to ply; vedo data/290.ply

# check on python2 the same stuff is ok
# cd ~/Projects/vedo/
# sudo -H pip install . -U
# python ~/Projects/vedo/examples/tutorial.py
# python ~/Dropbox/documents/ExamenesMedicos/RESONANCIA.py

# check notebooks:
# cd ~/Projects/vedo/notebooks/
# jupyter notebook > /dev/null 2>&1

# cd ~/Projects/vedo/
# rm -rf examples/*/.ipynb_checkpoints examples/*/*/.ipynb_checkpoints .ipynb_checkpoints/
# rm -rf examples/other/dolfin/navier_stokes_cylinder/ examples/other/dolfin/shuttle.xml
# rm examples/other/trimesh/featuretype.STL examples/other/trimesh/machinist.XAML
# rm examples/other/scene.npy examples/other/timecourse1d.npy vedo/data/290.ply
# rm examples/other/embryo.html examples/other/embryo.x3d notebooks/volumetric/.ipynb_checkpoints/

# git status
# git add [files]
# git commit -a -m 'comment'
# git push

# (sudo apt install twine)
# (python -m pip install --user --upgrade twine)
# python setup.py sdist bdist_wheel
# twine upload dist/vedo-?.?.?.tar.gz -r pypi

# make release

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
# scp -r build/html/* musy@pcba-sharpe012.embl.es:Projects/StagingServer/var/www/html/vtkplotter.embl.es/
# cd ~/Projects/StagingServer/var/www/html/vtkplotter.embl.es
# chmod -R 755 content
# comment out defs in docs.py

# version bump vedo/version.py

## to generate gif: ezgif.com


######################## fenics 2019.2 docker:

# To copy files over:
# sudo docker ps   # to know the container name
# cd Projects
# sudo docker cp  vedo admiring_panini:/home/fenics/shared/
# sudo chmod -R 755 vedo*/
#
# cd vedo
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
#    pip3 install vedo # OR
#    git clone https://github.com/marcomusy/vedo.git
#    cd vedo
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


























