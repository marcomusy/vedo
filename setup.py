from setuptools import setup

setup(
    name='vtkplotter',
    version='8.7.7', # change also in vtkplotter/__init__.py and docs/source/conf.py
    packages=['vtkplotter'],
    scripts=['bin/vtkplotter', 'bin/vtkconvert'],
    install_requires=['vtk', 'numpy'],
    description='A module to draw and analyse 3D and point clouds with VTK.',
    long_description="""A module to draw and analyse 3D and point clouds with VTK.
    Check out https://vtkplotter.embl.es for documentation.""",
    author='Marco Musy',
    author_email='marco.musy@gmail.com',
    license='MIT',
    url='https://github.com/marcomusy/vtkplotter',
    keywords='vtk 3D visualization mesh',
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
# cd ~/Projects/vtkplotter/
# pip install .
# cd examples
# ./run_all.sh
# vtkplotter            examples/data/embryo.tif
# vtkplotter -g -c blue examples/data/embryo.slc
# vtkplotter --slicer   examples/data/embryo.slc
# vtkplotter -s -c k    examples/data/timecourse/*vtk
# vtkplotter -n -c k    examples/data/timecourse/reference_3*

# # check version number here and in vtkplotter/__init__.py

# git status
# (sudo apt install twine)
# (python -m pip install --user --upgrade twine)
# python setup.py sdist bdist_wheel
# twine upload dist/vtkplotter-?.?.?.tar.gz -r pypi

# git status
# git commit -a -m 'comment'
# git push

# pip:
# # https://pypi.org/project/vtkplotter
# git:
# # check status at  https://github.com/marcomusy/vtkplotter

## to generate gif: ezgif.com

## to generate documentation:
# Install the dependencies in docs/requirements.txt
#  pip install -r docs/requirements.txt
# Run the documentaion generation:
#  cd docs
#  make html
# Open the HTML webpage
#  build/html/index.html
#
## Old pdoc:
# cd ~
# pdoc --overwrite --html vtkplotter
# mount_staging
# cp vtkplotter/* ~/Projects/StagingServer/var/www/html/vtkplotter.embl.es
# remove trig entries in index.htm
#



