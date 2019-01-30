"""
.. role:: raw-html-m2r(raw)
   :format: html


.. note:: **Please check out the** `git repository <https://github.com/marcomusy/vtkplotter>`_.

    A full list of examples can be found in directories:
        
    - `examples/basic <https://github.com/marcomusy/vtkplotter/blob/master/examples/basic>`_ ,
    - `examples/advanced <https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced>`_ ,
    - `examples/volumetric <https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric>`_,
    - `examples/others <https://github.com/marcomusy/vtkplotter/blob/master/examples/other>`_.
       
:raw-html-m2r:`<br />`

.. image:: https://user-images.githubusercontent.com/32848391/51558920-ec436e00-1e80-11e9-9d96-aa9b7c72d58b.png

:raw-html-m2r:`<br />`
:raw-html-m2r:`<br />`

"""

def _tips():
    import vtk, sys
    from vtkplotter import colors, __version__
    vvers = ' vtkplotter '+__version__+', vtk '+vtk.vtkVersion().GetVTKVersion()
    vvers += ', python ' + \
        str(sys.version_info[0])+'.'+str(sys.version_info[1])+' '
    n = len(vvers)
    if not colors._terminal_has_colors: n = 0
    colors.printc(' '*n+'_'*(59-n), c='blue')
    colors.printc(vvers, invert=1, dim=1, c='blue', end='')
    msg = ' '*(59-n)+'|\n' + '|'+' '*58+'|\n|Press:'
    msg += "\ti     to print info about selected object          |\n"
    msg += "|\tm     to minimise opacity of selected mesh         |\n"
    msg += "|\t.,    to reduce/increase opacity                   |\n"
    msg += "|\t/     to maximize opacity                          |\n"
    msg += "|\tw/s   to toggle wireframe/solid style              |\n"
    msg += "|\tp/P   to change point size of vertices             |\n"
    msg += "|\tl/L   to change edge line width                    |\n"
    msg += "|\tx     to toggle mesh visibility                    |\n"
    msg += "|\tX     to pop up a cutter widget tool               |\n"
    msg += "|\t1-3   to change mesh color                         |\n"
    msg += "|\tk/K   to show point/cell scalars as color          |\n"
    msg += "|\tn     to show surface mesh normals                 |\n"
    msg += "|\tC     to print current camera info                 |\n"
    msg += "|\tS     to save a screenshot                         |\n"
    msg += "|\tq/e   to continue/close the rendering window       |\n"
    msg += "|\tEsc   to exit program                              |\n"
    msg += '|'+'_'*58+'|\n'
    colors.printc(msg, c='blue')


_defs="""
.. |tutorial.py| replace:: tutorial.py
.. _tutorial.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/tutorial.py
.. |tutorial_subdivide| image:: https://user-images.githubusercontent.com/32848391/46819341-ca1b5980-cd83-11e8-97b7-12b053d76aac.png
.. |tutorial_spline| image:: https://user-images.githubusercontent.com/32848391/35976041-15781de8-0cdf-11e8-997f-aeb725bc33cc.png
    :width: 250 px
    :target: tutorial.py_
    :alt: tutorial.py

.. |thinplate_grid.py| replace:: thinplate_grid.py
.. _thinplate_grid.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/thinplate_grid.py
.. |thinplate_grid| image:: https://user-images.githubusercontent.com/32848391/51433540-d188b380-1c4c-11e9-81e7-a1cf4642c54b.png
    :width: 250 px
    :target: thinplate_grid.py_
    :alt: thinplate_grid.py
        
.. |gyroscope2.py| replace:: gyroscope2.py
.. _gyroscope2.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gyroscope2.py
.. |gyroscope2| image:: https://user-images.githubusercontent.com/32848391/50738942-687b5780-11d9-11e9-97f0-72bbd63f7d6e.gif
    :width: 250 px
    :target: gyroscope2.py_
    :alt: gyroscope2.py
        
.. |trail.py| replace:: trail.py
.. _trail.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/trail.py
.. |trail| image:: https://user-images.githubusercontent.com/32848391/50738846-be033480-11d8-11e9-99b7-c4ceb90ae482.jpg
    :width: 250 px
    :target: trail.py_
    :alt: trail.py

.. |fillholes.py| replace:: fillholes.py
.. _fillholes.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/fillholes.py
    :width: 250 px

.. |quadratic_morphing.py| replace:: quadratic_morphing.py
.. _quadratic_morphing.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/quadratic_morphing.py
.. |quadratic_morphing| image:: https://user-images.githubusercontent.com/32848391/50738890-db380300-11d8-11e9-9cef-4c1276cca334.jpg
    :width: 250 px
    :target: quadratic_morphing.py_
    :alt: quadratic_morphing.py

.. |align1.py| replace:: align1.py
.. _align1.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/align1.py
.. |align1| image:: https://user-images.githubusercontent.com/32848391/50738875-c196bb80-11d8-11e9-8bdc-b80fd01a928d.jpg
    :width: 250 px
    :target: align1.py_
    :alt: align1.py

.. |align2.py| replace:: align2.py
.. _align2.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/align2.py
.. |align2| image:: https://user-images.githubusercontent.com/32848391/50738874-c196bb80-11d8-11e9-9587-2177d1680b70.jpg
    :width: 250 px
    :target: align2.py_
    :alt: align2.py

.. |carcrash.py| replace:: carcrash.py
.. _carcrash.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/carcrash.py
.. |carcrash| image:: https://user-images.githubusercontent.com/32848391/50738869-c0fe2500-11d8-11e9-9b0f-c22c30050c34.jpg
    :width: 250 px
    :target: carcrash.py_
    :alt: carcrash.py

.. |mirror.py| replace:: mirror.py
.. _mirror.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/mirror.py
.. |mirror| image:: https://user-images.githubusercontent.com/32848391/50738855-bf346180-11d8-11e9-97a0-c9aaae6ce052.jpg
    :target: mirror.py_
    :alt: mirror.py

.. |shrink.py| replace:: shrink.py
.. _shrink.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/shrink.py
.. |shrink| image:: https://user-images.githubusercontent.com/32848391/46819143-41042280-cd83-11e8-9492-4f53679887fa.png
    :width: 250 px
    :target: shrink.py_
    :alt: shrink.py

.. |aspring.py| replace:: aspring.py
.. _aspring.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/aspring.py
.. |aspring| image:: https://user-images.githubusercontent.com/32848391/36788885-e97e80ae-1c8f-11e8-8b8f-ffc43dad1eb1.gif
    :width: 250 px
    :target: aspring.py_
    :alt: aspring.py

.. |delaunay2d.py| replace:: delaunay2d.py
.. _delaunay2d.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/delaunay2d.py
.. |delaunay2d| image:: https://user-images.githubusercontent.com/32848391/50738865-c0658e80-11d8-11e9-8616-b77363aa4695.jpg
    :width: 250 px
    :target: delaunay2d.py_
    :alt: delaunay2d.py

.. |moving_least_squares1D.py| replace:: moving_least_squares1D.py
.. _moving_least_squares1D.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares1D.py
.. |moving_least_squares1D| image:: https://user-images.githubusercontent.com/32848391/50738937-61544980-11d9-11e9-8be8-8826032b8baf.jpg
    :width: 250 px
    :target: moving_least_squares1D.py_
    :alt: moving_least_squares1D.py

.. |recosurface.py| replace:: recosurface.py
.. _recosurface.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/recosurface.py
.. |recosurface| image:: https://user-images.githubusercontent.com/32848391/46817107-b3263880-cd7e-11e8-985d-f5d158992f0c.png
    :target: recosurface.py_
    :alt: recosurface.py

.. |fatlimb.py| replace:: fatlimb.py
.. _fatlimb.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fatlimb.py
.. |fatlimb| image:: https://user-images.githubusercontent.com/32848391/50738945-7335ec80-11d9-11e9-9d3f-c6c19df8f10d.jpg
    :width: 250 px
    :target: fatlimb.py_
    :alt: fatlimb.py

.. |largestregion.py| replace:: largestregion.py
.. _largestregion.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/largestregion.py
    :width: 250 px
    :target: largestregion.py_
    :alt: largestregion.py

.. |fitplanes.py| replace:: fitplanes.py
.. _fitplanes.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitplanes.py

.. |mesh_coloring.py| replace:: mesh_coloring.py
.. _mesh_coloring.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/mesh_coloring.py
.. |mesh_coloring| image:: https://user-images.githubusercontent.com/32848391/50738856-bf346180-11d8-11e9-909c-a3f9d32c4e8c.jpg
    :width: 250 px
    :target: mesh_coloring.py_
    :alt: mesh_coloring.py

.. |mesh_alphas.py| replace:: mesh_alphas.py
.. _mesh_alphas.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/mesh_alphas.py
.. |mesh_alphas| image:: https://user-images.githubusercontent.com/32848391/50738857-bf346180-11d8-11e9-80a1-d283aed0b305.jpg
    :width: 250 px
    :target: mesh_alphas.py_
    :alt: mesh_alphas.py

.. |mesh_bands.py| replace:: mesh_bands.py
.. _mesh_bands.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/mesh_bands.py
.. |mesh_bands| image:: https://user-images.githubusercontent.com/32848391/51211548-26a78b00-1916-11e9-9306-67b677d1be3a.png
    :width: 250 px
    :target: mesh_bands.py_
    :alt: mesh_bands.py

.. |mesh_custom.py| replace:: mesh_custom.py
.. _mesh_custom.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/mesh_custom.py
.. |mesh_custom| image:: https://user-images.githubusercontent.com/32848391/51390972-20d9c180-1b31-11e9-955d-025f1ef24cb7.png
    :width: 250 px
    :target: mesh_custom.py_
    :alt: mesh_custom.py

.. |connVtx.py| replace:: connVtx.py
.. _connVtx.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/connVtx.py
.. |connVtx| image:: https://user-images.githubusercontent.com/32848391/51558919-ec436e00-1e80-11e9-91ac-0787c35fc20e.png
    :width: 250 px
    :target: connVtx.py_
    :alt: connVtx.py

.. |spherical_harmonics1.py| replace:: spherical_harmonics1.py
.. _spherical_harmonics1.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/spherical_harmonics1.py

.. |spherical_harmonics2.py| replace:: spherical_harmonics2.py
.. _spherical_harmonics2.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/spherical_harmonics2.py

.. |skeletonize.py| replace:: skeletonize.py
.. _skeletonize.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/skeletonize.py
.. |skeletonize| image:: https://user-images.githubusercontent.com/32848391/46820954-c5f13b00-cd87-11e8-87aa-286528a09de8.png
    :target: spherical_harmonics2.py_
    :alt: skeletonize.py

.. |gyroscope1.py| replace:: gyroscope1.py
.. _gyroscope1.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gyroscope1.py
.. |gyroscope1| image:: https://user-images.githubusercontent.com/32848391/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif
    :width: 250 px
    :target: gyroscope1.py_
    :alt: gyroscope1.py

.. |icon.py| replace:: icon.py
.. _icon.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/icon.py
.. |icon| image:: https://user-images.githubusercontent.com/32848391/50739009-2bfc2b80-11da-11e9-9e2e-a5e0e987a91a.jpg
    :width: 250 px
    :target: icon.py_

.. |lights.py| replace:: lights.py
.. _lights.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/lights.py
    :width: 250 px
    :target: lights.py_
    :alt: lights.py

.. |lorenz.py| replace:: lorenz.py
.. _lorenz.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/lorenz.py
.. |lorenz| image:: https://user-images.githubusercontent.com/32848391/46818115-be7a6380-cd80-11e8-8ffb-60af2631bf71.png
    :width: 250 px
    :target: lorenz.py_
    :alt: lorenz.py

.. |sliders.py| replace:: sliders.py
.. _sliders.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/sliders.py
.. |sliders| image:: https://user-images.githubusercontent.com/32848391/50738848-be033480-11d8-11e9-9b1a-c13105423a79.jpg
    :width: 250 px
    :target: sliders.py_
    :alt: sliders.py

.. |buttons.py| replace:: buttons.py
.. _buttons.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/buttons.py
.. |buttons| image:: https://user-images.githubusercontent.com/32848391/50738870-c0fe2500-11d8-11e9-9b78-92754f5c5968.jpg
    :width: 250 px
    :target: buttons.py_
    :alt: buttons.py

.. |cutter.py| replace:: cutter.py
.. _cutter.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/cutter.py
.. |cutter| image:: https://user-images.githubusercontent.com/32848391/50738866-c0658e80-11d8-11e9-955b-551d4d8b0db5.jpg
    :width: 250 px
    :target: cutter.py_
    :alt: cutter.py

.. |makeVideo.py| replace:: makeVideo.py
.. _makeVideo.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/other/makeVideo.py
.. |makeVideo| image:: https://user-images.githubusercontent.com/32848391/50739007-2bfc2b80-11da-11e9-97e6-620a3541a6fa.jpg
    :width: 250 px
    :target: makeVideo.py_
    :alt: makeVideo.py

.. |fitspheres1.py| replace:: fitspheres1.py
.. _fitspheres1.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitspheres1.py

.. |fitspheres2.py| replace:: fitspheres2.py
.. _fitspheres2.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitspheres2.py
.. |fitspheres2| image:: https://user-images.githubusercontent.com/32848391/50738943-687b5780-11d9-11e9-87a6-054e0fe76241.jpg
    :width: 250 px
    :target: fitspheres2.py_
    :alt: fitspheres2.py

.. |fxy.py| replace:: fxy.py
.. _fxy.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/fxy.py
.. |fxy| image:: https://user-images.githubusercontent.com/32848391/36611824-fd524fac-18d4-11e8-8c76-d3d1b1bb3954.png
    :width: 250 px
    :target: fxy.py_
    :alt: fxy.py

.. |histo2D.py| replace:: histo2D.py
.. _histo2D.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/histo2D.py
.. |histo2D| image:: https://user-images.githubusercontent.com/32848391/50738861-bfccf800-11d8-11e9-9698-c0b9dccdba4d.jpg
    :width: 250 px
    :alt: histo2D.py

.. |align3.py| replace:: align3.py
.. _align3.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/align3.py
.. |align3| image:: https://user-images.githubusercontent.com/32848391/50738873-c196bb80-11d8-11e9-8653-a41108a5f02d.png
    :width: 250 px
    :target: align3.py_
    :alt: align3.py

.. |fitline.py| replace:: fitline.py
.. _fitline.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/fitline.py
    :width: 250 px
    :target: fitline.py_
    :alt: fitline.py

.. |pca.py| replace:: pca.py
.. _pca.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/pca.py
.. |pca| image:: https://user-images.githubusercontent.com/32848391/50738852-be9bcb00-11d8-11e9-8ac8-ad9278d9cee0.jpg
    :width: 250 px
    :target: pca.py_
    :alt: pca.py

.. |cell_main.py| replace:: cell_main.py
.. _cell_main.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/cell_main.py
.. |cell_main| image:: https://user-images.githubusercontent.com/32848391/50738947-7335ec80-11d9-11e9-9a45-6053b4eaf9f9.jpg
    :width: 250 px
    :target: cell_main.py_
    :alt: cell_main.py

.. |mesh_smoothers.py| replace:: mesh_smoothers.py
.. _mesh_smoothers.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/mesh_smoothers.py
.. |mesh_smoothers| image:: https://user-images.githubusercontent.com/32848391/50738939-67e2c100-11d9-11e9-90cb-716ff3f03f67.jpg
    :width: 250 px
    :target: mesh_smoothers.py_
    :alt: mesh_smoothers.py

.. |moving_least_squares3D.py| replace:: moving_least_squares3D.py
.. _moving_least_squares3D.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares3D.py
.. |moving_least_squares3D| image:: https://user-images.githubusercontent.com/32848391/50738935-61544980-11d9-11e9-9c20-f2ce944d2238.jpg
    :width: 250 px
    :target: moving_least_squares3D.py_
    :alt: moving_least_squares3D.py

.. |moving_least_squares2D.py| replace:: moving_least_squares2D.py
.. _moving_least_squares2D.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/moving_least_squares2D.py
.. |moving_least_squares2D| image:: https://user-images.githubusercontent.com/32848391/50738936-61544980-11d9-11e9-9efb-e2a923762b72.jpg
    :width: 250 px
    :target: moving_least_squares2D.py_
    :alt: moving_least_squares2D.py

.. |boolean.py| replace:: boolean.py
.. _boolean.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/boolean.py
.. |boolean| image:: https://user-images.githubusercontent.com/32848391/50738871-c0fe2500-11d8-11e9-8812-442b69be6db9.png
    :width: 250 px
    :target: boolean.py_
    :alt: boolean.py

.. |surfIntersect.py| replace:: surfIntersect.py
.. _surfIntersect.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/surfIntersect.py

.. |probeLine.py| replace:: probeLine.py
.. _probeLine.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/probeLine.py
.. |probeLine| image:: https://user-images.githubusercontent.com/32848391/48198460-3aa0a080-e359-11e8-982d-23fadf4de66f.jpg
    :width: 250 px
    :target: probeLine.py_
    :alt: probeLine.py

.. |probePlane.py| replace:: probePlane.py
.. _probePlane.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/probePlane.py
.. |probePlane| image:: https://user-images.githubusercontent.com/32848391/48198461-3aa0a080-e359-11e8-8c29-18f287f105e6.jpg
    :width: 250 px
    :target: probePlane.py_
    :alt: probePlane.py

.. |imageOperations.py| replace:: imageOperations.py
.. _imageOperations.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/imageOperations.py
.. |imageOperations| image:: https://user-images.githubusercontent.com/32848391/48198940-d1ba2800-e35a-11e8-96a7-ffbff797f165.jpg
    :width: 250 px
    :alt: imageOperations.py

.. |clustering.py| replace:: clustering.py
.. _clustering.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/clustering.py
.. |clustering| image:: https://user-images.githubusercontent.com/32848391/46817286-2039ce00-cd7f-11e8-8b29-42925e03c974.png
    :width: 250 px
    :target: clustering.py_
    :alt: clustering.py

.. |thinplate.py| replace:: thinplate.py
.. _thinplate.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/thinplate.py
.. |thinplate| image:: https://user-images.githubusercontent.com/32848391/51403917-34495480-1b52-11e9-956c-918c7805a9b5.png
    :width: 250 px
    :target: thinplate.py_
    :alt: thinplate.py

.. |readStructuredPoints.py| replace:: readStructuredPoints.py
.. _readStructuredPoints.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/readStructuredPoints.py
.. |readStructuredPoints| image:: https://user-images.githubusercontent.com/32848391/48198462-3b393700-e359-11e8-8272-670bd5f2db42.jpg
    :width: 250 px
    :target: readStructuredPoints.py_
    :alt: readStructuredPoints.py

.. |buildpolydata.py| replace:: buildpolydata.py
.. _buildpolydata.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/buildpolydata.py
.. |buildpolydata| image:: https://user-images.githubusercontent.com/32848391/51032546-bf4dac00-15a0-11e9-9e1e-035fff9c05eb.png
    :width: 250 px
    :alt: buildpolydata.py

.. |colorcubes.py| replace:: colorcubes.py
.. _colorcubes.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/colorcubes.py
.. |colorcubes| image:: https://user-images.githubusercontent.com/32848391/50738867-c0658e80-11d8-11e9-9e05-ac69b546b7ec.png
    :width: 250 px
    :target: colorcubes.py_
    :alt: colorcubes.py

.. |colorpalette.py| replace:: colorpalette.py
.. _colorpalette.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/other/colorpalette.py
.. |colorpalette| image:: https://user-images.githubusercontent.com/32848391/50739011-2c94c200-11da-11e9-8f36-ede1b2a014a8.jpg
    :width: 250 px
    :target: colorpalette.py
    :alt: colorpalette.py

.. |colormaps| image:: https://user-images.githubusercontent.com/32848391/50738804-577e1680-11d8-11e9-929e-fca17a8ac6f3.jpg
    :width: 450 px
    :alt: colormaps

.. |tannerhelland| replace:: tannerhelland
.. _tannerhelland: http://www.tannerhelland.com/4435/convert-temperature-rgb-algorithm-code

.. |colorprint.py| replace:: colorprint.py
.. _colorprint.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/other/colorprint.py
.. |colorprint| image:: https://user-images.githubusercontent.com/32848391/50739010-2bfc2b80-11da-11e9-94de-011e50a86e61.jpg
    :target: colorprint.py_
    :alt: colorprint.py

.. |ribbon.py| replace:: ribbon.py
.. _ribbon.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/ribbon.py
.. |ribbon| image:: https://user-images.githubusercontent.com/32848391/50738851-be9bcb00-11d8-11e9-80ee-bd73c1c29c06.jpg
    :width: 250 px
    :target: ribbon.py_
    :alt: ribbon.py

.. |manyspheres.py| replace:: manyspheres.py
.. _manyspheres.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/manyspheres.py
.. |manyspheres| image:: https://user-images.githubusercontent.com/32848391/46818673-1f566b80-cd82-11e8-9a61-be6a56160f1c.png
    :target: manyspheres.py_

.. |earth.py| replace:: earth.py
.. _earth.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/earth.py
.. |earth| image:: https://user-images.githubusercontent.com/32848391/51031592-5a448700-159d-11e9-9b66-bee6abb18679.png
    :width: 250 px
    :target: earth.py_
    :alt: earth.py

.. |brownian2D.py| replace:: brownian2D.py
.. _brownian2D.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/brownian2D.py
.. |brownian2D| image:: https://user-images.githubusercontent.com/32848391/50738948-73ce8300-11d9-11e9-8ef6-fc4f64c4a9ce.gif
    :width: 250 px
    :target: brownian2D.py_

.. |gas.py| replace:: gas.py
.. _gas.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/gas.py
.. |gas| image:: https://user-images.githubusercontent.com/32848391/50738954-7e891800-11d9-11e9-95aa-67c92ca6476b.gif
    :width: 250 px
    :target: gas.py_
    :alt: gas.py

.. |tube.py| replace:: tube.py
.. _tube.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/tube.py
.. |tube| image:: https://user-images.githubusercontent.com/32848391/51801626-adc30000-2240-11e9-8866-9d9d5d8790ab.png
    :width: 250 px
    :target: tube.py_
    :alt: tube.py

.. |mesh_threshold.py| replace:: mesh_threshold.py
.. _mesh_threshold.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/mesh_threshold.py
.. |mesh_threshold| image:: https://user-images.githubusercontent.com/32848391/51807663-4762cf80-228a-11e9-9d0c-184bb11a97bf.png
    :width: 300 px
    :target: mesh_threshold.py_
    :alt: mesh_threshold.py

.. |cutWithMesh.py| replace:: cutWithMesh.py
.. _cutWithMesh.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/cutWithMesh.py
.. |cutWithMesh| image:: https://user-images.githubusercontent.com/32848391/51808907-e7c0f000-229a-11e9-98a7-fefc7261b3c3.png
    :width: 300 px
    :target: cutWithMesh.py_
    :alt: cutWithMesh.py

.. |paraboloid| image:: https://user-images.githubusercontent.com/32848391/51211547-260ef480-1916-11e9-95f6-4a677e37e355.png
    :width: 250 px
    :alt: paraboloid

.. |isosurfaces.py| replace:: isosurfaces.py
.. _isosurfaces.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/isosurfaces.py
.. |isosurfaces| image:: https://user-images.githubusercontent.com/32848391/51558920-ec436e00-1e80-11e9-9d96-aa9b7c72d58b.png
    :width: 250 px
    :target: isosurfaces.py_
    :alt: isosurfaces.py

.. |meshquality.py| replace:: meshquality.py
.. _meshquality.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/meshquality.py
.. |meshquality| image:: https://user-images.githubusercontent.com/32848391/51831269-fb4b7580-22f1-11e9-81ea-13467a5649ca.png
    :width: 250 px
    :target: meshquality.py_
    :alt: meshquality.py

.. |geodesic.py| replace:: geodesic.py
.. _geodesic.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/geodesic.py
.. |geodesic| image:: https://user-images.githubusercontent.com/32848391/51855637-015f4780-232e-11e9-92ca-053a558e7f70.png
    :width: 250 px
    :target: geodesic.py_
    :alt: geodesic.py
    
 
.. |cutAndCap.py| replace:: cutAndCap.py
.. _cutAndCap.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/cutAndCap.py
.. |cutAndCap| image:: https://user-images.githubusercontent.com/32848391/51930515-16ee7300-23fb-11e9-91af-2b6b3d626246.png
    :width: 250 px
    :target: cutAndCap.py_
    :alt: cutAndCap.py


.. |convexHull.py| replace:: convexHull.py
.. _convexHull.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/advanced/convexHull.py
.. |convexHull| image:: https://user-images.githubusercontent.com/32848391/51932732-068cc700-2400-11e9-9b68-30294a4fa4e3.png
    :width: 250 px
    :target: convexHull.py_
    :alt: convexHull.py

.. |curvature| image:: https://user-images.githubusercontent.com/32848391/51934810-c2e88c00-2404-11e9-8e7e-ca0b7984bbb7.png
    :alt: curvature

.. |progbar| image:: https://user-images.githubusercontent.com/32848391/51858823-ed1f4880-2335-11e9-8788-2d102ace2578.png
    :alt: progressbar

.. |multiwindows| image:: https://user-images.githubusercontent.com/32848391/50738853-be9bcb00-11d8-11e9-9c8e-69864ad7c045.jpg
    :alt: multiwindows

.. |annotations.py| replace:: annotations.py
.. _annotations.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/basic/annotations.py

.. |mesh2volume.py| replace:: mesh2volume.py
.. _mesh2volume.py: https://github.com/marcomusy/vtkplotter/blob/master/examples/volumetric/mesh2volume.py


"""










