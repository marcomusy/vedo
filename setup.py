from setuptools import setup

setup(
    name='vtkplotter',
    version='8.4.0', #change also in __init__.py
    packages=['vtkplotter'],
    scripts=['bin/vtkplotter'],
    install_requires=[], # vtk and numpy are needed but better install it manually
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
# source run_all.sh

# # check version number here and in plotter.py

# ana3
# git status
# python -m pip install --user --upgrade twine
# python setup.py sdist bdist_wheel

# git commit -a -m 'comment'
# git push
# twine upload dist/vtkplotter-?.?.?.tar.gz -r pypi

# # check status at  https://pypi.org/project/vtkplotter/
# # check status at  https://github.com/marcomusy/vtkplotter
