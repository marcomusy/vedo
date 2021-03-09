.. role:: raw-html-m2r(raw)
   :format: html

.. image:: https://user-images.githubusercontent.com/32848391/110344277-9bc20700-802d-11eb-8c0d-2e97226a9a32.png
   :target: https://vedo.embl.es

:raw-html-m2r:`<br />`

.. image:: https://pepy.tech/badge/vtkplotter
   :target: https://pepy.tech/project/vtkplotter
   :alt: Downloads

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://en.wikipedia.org/wiki/MIT_License
   :alt: lics

.. image:: https://img.shields.io/badge/python-2.7%7C3.6-brightgreen.svg
   :target: https://pypi.org/project/vedo
   :alt: pythvers

.. image:: https://img.shields.io/badge/docs%20by-gendocs-blue.svg
   :target: https://gendocs.readthedocs.io/en/latest/
   :alt: Documentation Built by gendocs

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.2561402.svg
   :target: https://doi.org/10.5281/zenodo.2561402

---------------------

`vedo` is a fast and lightweight python module
for scientific analysis and visualization of 3d objects.

Not limited to 3d, `vedo` can also generate high quality renderings in 2d (scatter plots, histograms etc).

`Check out the project web page <https://vedo.embl.es>`_.

Philosophy
----------

Inspired by the `vpython <https://vpython.org/>`_ *manifesto* "3D programming for ordinary mortals",
*vedo* makes it easy to work wth three-dimensional objects, create displays and animations
in just a few lines of code, even for those with less programming experience.

`vedo` is based on `VTK <https://www.vtk.org/>`_ and `numpy <http://www.numpy.org/>`_,
with no other dependencies.


Download and Install:
---------------------

.. code-block:: bash

   pip install vedo

Check out the **Git repository** here: https://github.com/marcomusy/vedo

*Windows-10 users* can manually place this file
`vedo.bat <https://github.com/marcomusy/vedo/blob/master/vedo.bat>`_
on the desktop to *drag&drop* files to visualize.
(Need to edit the path of their local python installation).


Examples
--------

Run any of the available scripts from with:

.. code-block:: bash

    vedo --list
    vedo -r covid19


More than 300 examples are sorted by subject in directories:


.. image:: https://vedo.embl.es/images/logos/bar.png
   :target: https://vedo.embl.es
   :alt: vedo



Mesh format conversion
^^^^^^^^^^^^^^^^^^^^^^

The command ``vedo-convert`` can be used to convert multiple files from a format to a different one:

.. code-block:: bash

   Usage: vedo-convert [-h] [-to] [files [files ...]]
   allowed targets formats: [vtk, vtp, vtu, vts, ply, stl, byu, xml]

   Example: > vedo-convert myfile.vtk -to ply
