from setuptools import setup

setup(
    name='vtkplotter',
    version='8.1',
    packages=['vtkplotter'],
    scripts=['bin/vtkplotter'],
    install_requires=[], # vtk and numpy are needed but better install it manually
    description='A helper class to easily draw 3D objects',
    long_description='A helper class to easily draw 3D objects',
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