from setuptools import setup

setup(
    name='vtkplotter',
    version='8.5.7',     # change also in vtkplotter/__init__.py
    packages=['vtkplotter'],
    scripts=['bin/vtkplotter', 'bin/vtkconvert'],
    install_requires=[], # vtk and numpy are needed but better install manually
    description='A helper class to easily draw and analyse 3D shapes',
    long_description="""A helper class to easily draw and analyse 3D shapes.
    Check out https://github.com/marcomusy/vtkplotter for documentation.""",
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
                'Topic :: Scientific/Engineering :: Information Analysis'],
    include_package_data=True
)

##############################################################
# # check examples
# cd ~/Projects/vtkplotter/
# pip install .
# cd examples
# ./run_all.sh

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
# # https://pepy.tech/project/vtkplotter # downloads
# git:
# # check status at  https://github.com/marcomusy/vtkplotter

## to generate gif: ezgif.com

## to generate documentation:
# cd ~
# pdoc --overwrite --html vtkplotter
# mount_staging
# cp vtkplotter/* ~/Projects/StagingServer/var/www/html/vtkplotter.embl.es