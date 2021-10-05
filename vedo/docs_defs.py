#!/usr/bin/env python3
# -*- coding: utf-8 -*-

_substitutions_defs = """

.. |gyroscope2.py| replace:: gyroscope2.py
.. _gyroscope2.py: https://github.com/marcomusy/vedo/tree/master/examples/simulations/gyroscope2.py
.. |gyroscope2| image:: https://user-images.githubusercontent.com/32848391/50738942-687b5780-11d9-11e9-97f0-72bbd63f7d6e.gif
    :width: 200 px
    :target: gyroscope2.py_
    :alt: gyroscope2.py

.. |trail.py| replace:: trail.py
.. _trail.py: https://github.com/marcomusy/vedo/tree/master/examples/simulations/trail.py
.. |trail| image:: https://user-images.githubusercontent.com/32848391/58370826-4aee2680-7f0b-11e9-91e6-3120770cfede.gif
    :width: 200 px
    :target: trail.py_
    :alt: trail.py

.. |fillholes.py| replace:: fillholes.py
.. _fillholes.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/fillholes.py

.. |quadratic_morphing.py| replace:: quadratic_morphing.py
.. _quadratic_morphing.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/quadratic_morphing.py
.. |quadratic_morphing| image:: https://user-images.githubusercontent.com/32848391/50738890-db380300-11d8-11e9-9cef-4c1276cca334.jpg
    :width: 200 px
    :target: quadratic_morphing.py_
    :alt: quadratic_morphing.py

.. |align1.py| replace:: align1.py
.. _align1.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/align1.py
.. |align1| image:: https://user-images.githubusercontent.com/32848391/50738875-c196bb80-11d8-11e9-8bdc-b80fd01a928d.jpg
    :width: 200 px
    :target: align1.py_
    :alt: align1.py

.. |align2.py| replace:: align2.py
.. _align2.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/align2.py
.. |align2| image:: https://user-images.githubusercontent.com/32848391/50738874-c196bb80-11d8-11e9-9587-2177d1680b70.jpg
    :width: 200 px
    :target: align2.py_
    :alt: align2.py

.. |mirror.py| replace:: mirror.py
.. _mirror.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/mirror.py
.. |mirror| image:: https://user-images.githubusercontent.com/32848391/50738855-bf346180-11d8-11e9-97a0-c9aaae6ce052.jpg
    :target: mirror.py_
    :alt: mirror.py

.. |shrink.py| replace:: shrink.py
.. _shrink.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/shrink.py
.. |shrink| image:: https://user-images.githubusercontent.com/32848391/46819143-41042280-cd83-11e8-9492-4f53679887fa.png
    :width: 200 px
    :target: shrink.py_
    :alt: shrink.py

.. |aspring.py| replace:: aspring.py
.. _aspring.py: https://github.com/marcomusy/vedo/tree/master/examples/simulations/aspring.py
.. |aspring| image:: https://user-images.githubusercontent.com/32848391/36788885-e97e80ae-1c8f-11e8-8b8f-ffc43dad1eb1.gif
    :width: 200 px
    :target: aspring.py_
    :alt: aspring.py

.. |delaunay2d.py| replace:: delaunay2d.py
.. _delaunay2d.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/delaunay2d.py
.. |delaunay2d| image:: https://user-images.githubusercontent.com/32848391/50738865-c0658e80-11d8-11e9-8616-b77363aa4695.jpg
    :width: 200 px
    :target: delaunay2d.py_
    :alt: delaunay2d.py

.. |moving_least_squares1D.py| replace:: moving_least_squares1D.py
.. _moving_least_squares1D.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/moving_least_squares1D.py
.. |moving_least_squares1D| image:: https://user-images.githubusercontent.com/32848391/50738937-61544980-11d9-11e9-8be8-8826032b8baf.jpg
    :width: 200 px
    :target: moving_least_squares1D.py_
    :alt: moving_least_squares1D.py

.. |recosurface.py| replace:: recosurface.py
.. _recosurface.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/recosurface.py
.. |recosurface| image:: https://user-images.githubusercontent.com/32848391/46817107-b3263880-cd7e-11e8-985d-f5d158992f0c.png
    :target: recosurface.py_
    :alt: recosurface.py

.. |fatlimb.py| replace:: fatlimb.py
.. _fatlimb.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/fatlimb.py
.. |fatlimb| image:: https://user-images.githubusercontent.com/32848391/50738945-7335ec80-11d9-11e9-9d3f-c6c19df8f10d.jpg
    :width: 200 px
    :target: fatlimb.py_
    :alt: fatlimb.py

.. |largestregion.py| replace:: largestregion.py
.. _largestregion.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/largestregion.py

.. |fitplanes.py| replace:: fitplanes.py
.. _fitplanes.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/fitplanes.py

.. |mesh_coloring.py| replace:: mesh_coloring.py
.. _mesh_coloring.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_coloring.py
.. |mesh_coloring| image:: https://user-images.githubusercontent.com/32848391/50738856-bf346180-11d8-11e9-909c-a3f9d32c4e8c.jpg
    :width: 200 px
    :target: mesh_coloring.py_
    :alt: mesh_coloring.py

.. |mesh_alphas.py| replace:: mesh_alphas.py
.. _mesh_alphas.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_alphas.py
.. |mesh_alphas| image:: https://user-images.githubusercontent.com/32848391/50738857-bf346180-11d8-11e9-80a1-d283aed0b305.jpg
    :width: 200 px
    :target: mesh_alphas.py_
    :alt: mesh_alphas.py

.. |mesh_custom.py| replace:: mesh_custom.py
.. _mesh_custom.py: https://github.com/marcomusy/vedo/tree/master/examples/mesh_custom.py
.. |mesh_custom| image:: https://user-images.githubusercontent.com/32848391/51390972-20d9c180-1b31-11e9-955d-025f1ef24cb7.png
    :width: 200 px
    :target: mesh_custom.py_
    :alt: mesh_custom.py

.. |connVtx.py| replace:: connVtx.py
.. _connVtx.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/connVtx.py
.. |connVtx| image:: https://user-images.githubusercontent.com/32848391/51558919-ec436e00-1e80-11e9-91ac-0787c35fc20e.png
    :width: 200 px
    :target: connVtx.py_
    :alt: connVtx.py

.. |spherical_harmonics1.py| replace:: spherical_harmonics1.py
.. _spherical_harmonics1.py: https://github.com/marcomusy/vedo/tree/master/examples/other/spherical_harmonics1.py

.. |spherical_harmonics2.py| replace:: spherical_harmonics2.py
.. _spherical_harmonics2.py: https://github.com/marcomusy/vedo/tree/master/examples/other/spherical_harmonics2.py

.. |skeletonize.py| replace:: skeletonize.py
.. _skeletonize.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/skeletonize.py
.. |skeletonize| image:: https://user-images.githubusercontent.com/32848391/46820954-c5f13b00-cd87-11e8-87aa-286528a09de8.png
    :target: spherical_harmonics2.py_
    :alt: skeletonize.py

.. |gyroscope1.py| replace:: gyroscope1.py
.. _gyroscope1.py: https://github.com/marcomusy/vedo/tree/master/examples/simulations/gyroscope1.py
.. |gyroscope1| image:: https://user-images.githubusercontent.com/32848391/39766016-85c1c1d6-52e3-11e8-8575-d167b7ce5217.gif
    :width: 200 px
    :target: gyroscope1.py_
    :alt: gyroscope1.py

.. |icon.py| replace:: icon.py
.. _icon.py: https://github.com/marcomusy/vedo/tree/master/examples/other/icon.py
.. |icon| image:: https://user-images.githubusercontent.com/32848391/50739009-2bfc2b80-11da-11e9-9e2e-a5e0e987a91a.jpg
    :width: 200 px
    :target: icon.py_

.. |lights.py| replace:: lights.py
.. _lights.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/lights.py
    :width: 200 px
    :target: lights.py_
    :alt: lights.py

.. |lorenz.py| replace:: lorenz.py
.. _lorenz.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/lorenz.py
.. |lorenz| image:: https://user-images.githubusercontent.com/32848391/46818115-be7a6380-cd80-11e8-8ffb-60af2631bf71.png
    :width: 200 px
    :target: lorenz.py_
    :alt: lorenz.py

.. |sliders1.py| replace:: sliders1.py
.. _sliders1.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/sliders1.py
.. |sliders1| image:: https://user-images.githubusercontent.com/32848391/50738848-be033480-11d8-11e9-9b1a-c13105423a79.jpg
    :width: 200 px
    :target: sliders1.py_
    :alt: sliders1.py

.. |sliders2.py| replace:: sliders2.py
.. _sliders2.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/sliders2.py

.. |buttons.py| replace:: buttons.py
.. _buttons.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/buttons.py
.. |buttons| image:: https://user-images.githubusercontent.com/32848391/50738870-c0fe2500-11d8-11e9-9b78-92754f5c5968.jpg
    :width: 200 px
    :target: buttons.py_
    :alt: buttons.py

.. |cutter.py| replace:: cutter.py
.. _cutter.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/cutter.py
.. |cutter| image:: https://user-images.githubusercontent.com/32848391/50738866-c0658e80-11d8-11e9-955b-551d4d8b0db5.jpg
    :width: 200 px
    :target: cutter.py_
    :alt: cutter.py

.. |makeVideo.py| replace:: makeVideo.py
.. _makeVideo.py: https://github.com/marcomusy/vedo/tree/master/examples/other/makeVideo.py
.. |makeVideo| image:: https://user-images.githubusercontent.com/32848391/50739007-2bfc2b80-11da-11e9-97e6-620a3541a6fa.jpg
    :width: 200 px
    :target: makeVideo.py_
    :alt: makeVideo.py

.. |fitspheres1.py| replace:: fitspheres1.py
.. _fitspheres1.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/fitspheres1.py

.. |fitspheres2.py| replace:: fitspheres2.py
.. _fitspheres2.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/fitspheres2.py
.. |fitspheres2| image:: https://user-images.githubusercontent.com/32848391/50738943-687b5780-11d9-11e9-87a6-054e0fe76241.jpg
    :width: 200 px
    :target: fitspheres2.py_
    :alt: fitspheres2.py

.. |plot_fxy.py| replace:: plot_fxy.py
.. _plot_fxy.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_fxy.py
.. |plot_fxy| image:: https://user-images.githubusercontent.com/32848391/36611824-fd524fac-18d4-11e8-8c76-d3d1b1bb3954.png
    :width: 200 px
    :target: plot_fxy.py_
    :alt: plot_fxy.py

.. |histo_hexagonal.py| replace:: histo_hexagonal.py
.. _histo_hexagonal.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_hexagonal.py
.. |histo_hexagonal| image:: https://user-images.githubusercontent.com/32848391/72434748-b471bc80-379c-11ea-95d7-d70333770582.png
    :width: 200 px
    :target: histo_hexagonal.py_
    :alt: histo_hexagonal.py

.. |histo_1D.py| replace:: histo_1D.py
.. _histo_1D.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_1D.py
.. |histo_1D| image:: https://user-images.githubusercontent.com/32848391/68141260-77cc4e00-ff2d-11e9-9280-0efc5b87314d.png
    :width: 200 px
    :target: histo_1D.py_
    :alt: histo_1D.py

.. |histo_violin.py| replace:: histo_violin.py
.. _histo_violin.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_violin.py
.. |histo_violin| image:: https://user-images.githubusercontent.com/32848391/73481240-b55d3d80-439b-11ea-89a4-6c35ecc84b0d.png
    :width: 200 px
    :target: histo_violin.py_
    :alt: histo_violin.py

.. |align3.py| replace:: align3.py
.. _align3.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/align3.py
.. |align3| image:: https://user-images.githubusercontent.com/32848391/50738873-c196bb80-11d8-11e9-8653-a41108a5f02d.png
    :width: 200 px
    :target: align3.py_
    :alt: align3.py

.. |pca.py| replace:: pca.py
.. _pca.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/pca.py
.. |pca| image:: https://user-images.githubusercontent.com/32848391/50738852-be9bcb00-11d8-11e9-8ac8-ad9278d9cee0.jpg
    :width: 200 px
    :target: pca.py_
    :alt: pca.py

.. |cell_colony.py| replace:: cell_colony.py
.. _cell_colony.py: https://github.com/marcomusy/vedo/tree/master/examples/simulations/cell_colony.py
.. |cell_colony| image:: https://user-images.githubusercontent.com/32848391/50738947-7335ec80-11d9-11e9-9a45-6053b4eaf9f9.jpg
    :width: 200 px
    :target: cell_colony.py_
    :alt: cell_colony.py

.. |mesh_smoother1.py| replace:: mesh_smoother1.py
.. _mesh_smoother1.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/mesh_smoother1.py
.. |mesh_smoother1| image:: https://user-images.githubusercontent.com/32848391/50738939-67e2c100-11d9-11e9-90cb-716ff3f03f67.jpg
    :width: 200 px
    :target: mesh_smoother1.py_
    :alt: mesh_smoother1.py

.. |moving_least_squares3D.py| replace:: moving_least_squares3D.py
.. _moving_least_squares3D.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/moving_least_squares3D.py
.. |moving_least_squares3D| image:: https://user-images.githubusercontent.com/32848391/50738935-61544980-11d9-11e9-9c20-f2ce944d2238.jpg
    :width: 200 px
    :target: moving_least_squares3D.py_
    :alt: moving_least_squares3D.py

.. |moving_least_squares2D.py| replace:: moving_least_squares2D.py
.. _moving_least_squares2D.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/moving_least_squares2D.py
.. |moving_least_squares2D| image:: https://user-images.githubusercontent.com/32848391/50738936-61544980-11d9-11e9-9efb-e2a923762b72.jpg
    :width: 200 px
    :target: moving_least_squares2D.py_
    :alt: moving_least_squares2D.py

.. |boolean.py| replace:: boolean.py
.. _boolean.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/boolean.py
.. |boolean| image:: https://user-images.githubusercontent.com/32848391/50738871-c0fe2500-11d8-11e9-8812-442b69be6db9.png
    :width: 200 px
    :target: boolean.py_
    :alt: boolean.py

.. |surfIntersect.py| replace:: surfIntersect.py
.. _surfIntersect.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/surfIntersect.py

.. |probeLine1.py| replace:: probeLine1.py
.. _probeLine1.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/probeLine1.py
.. |probeLine1| image:: https://user-images.githubusercontent.com/32848391/48198460-3aa0a080-e359-11e8-982d-23fadf4de66f.jpg
    :width: 200 px
    :target: probeLine1.py_
    :alt: probeLine1.py

.. |probeLine2.py| replace:: probeLine2.py
.. _probeLine2.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/probeLine2.py

.. |slicePlane1.py| replace:: slicePlane1.py
.. _slicePlane1.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/slicePlane1.py
.. |slicePlane1| image:: https://user-images.githubusercontent.com/32848391/48198461-3aa0a080-e359-11e8-8c29-18f287f105e6.jpg
    :width: 200 px
    :target: slicePlane1.py_
    :alt: slicePlane1.py

.. |volumeOperations.py| replace:: volumeOperations.py
.. _volumeOperations.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/volumeOperations.py
.. |volumeOperations| image:: https://user-images.githubusercontent.com/32848391/48198940-d1ba2800-e35a-11e8-96a7-ffbff797f165.jpg
    :width: 200 px
    :alt: volumeOperations.py

.. |clustering.py| replace:: clustering.py
.. _clustering.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/clustering.py
.. |clustering| image:: https://user-images.githubusercontent.com/32848391/46817286-2039ce00-cd7f-11e8-8b29-42925e03c974.png
    :width: 200 px
    :target: clustering.py_
    :alt: clustering.py

.. |warp1.py| replace:: warp1.py
.. _warp1.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp1.py
.. |warp1| image:: https://user-images.githubusercontent.com/32848391/51403917-34495480-1b52-11e9-956c-918c7805a9b5.png
    :width: 200 px
    :target: warp1.py_
    :alt: warp1.py

.. |colorcubes.py| replace:: colorcubes.py
.. _colorcubes.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/colorcubes.py
.. |colorcubes| image:: https://user-images.githubusercontent.com/32848391/50738867-c0658e80-11d8-11e9-9e05-ac69b546b7ec.png
    :width: 200 px
    :target: colorcubes.py_
    :alt: colorcubes.py

.. |colorpalette.py| replace:: colorpalette.py
.. _colorpalette.py: https://github.com/marcomusy/vedo/tree/master/examples/other/colorpalette.py
.. |colorpalette| image:: https://user-images.githubusercontent.com/32848391/50739011-2c94c200-11da-11e9-8f36-ede1b2a014a8.jpg
    :width: 200 px
    :target: colorpalette.py
    :alt: colorpalette.py

.. |colormaps| image:: https://user-images.githubusercontent.com/32848391/50738804-577e1680-11d8-11e9-929e-fca17a8ac6f3.jpg
    :width: 450 px
    :alt: colormaps

.. |colorprint.py| replace:: printc.py
.. _colorprint.py: https://github.com/marcomusy/vedo/tree/master/examples/other/printc.py
.. |colorprint| image:: https://user-images.githubusercontent.com/32848391/50739010-2bfc2b80-11da-11e9-94de-011e50a86e61.jpg
    :target: colorprint.py_
    :alt: colorprint.py

.. |ribbon.py| replace:: ribbon.py
.. _ribbon.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/ribbon.py
.. |ribbon| image:: https://user-images.githubusercontent.com/32848391/50738851-be9bcb00-11d8-11e9-80ee-bd73c1c29c06.jpg
    :width: 200 px
    :target: ribbon.py_
    :alt: ribbon.py

.. |manyspheres.py| replace:: manyspheres.py
.. _manyspheres.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/manyspheres.py
.. |manyspheres| image:: https://user-images.githubusercontent.com/32848391/46818673-1f566b80-cd82-11e8-9a61-be6a56160f1c.png
    :target: manyspheres.py_

.. |manypoints.py| replace:: manypoints.py
.. _manypoints.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/manypoints.py

.. |earth.py| replace:: earth.py
.. _earth.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/earth.py
.. |earth| image:: https://user-images.githubusercontent.com/32848391/51031592-5a448700-159d-11e9-9b66-bee6abb18679.png
    :width: 200 px
    :target: earth.py_
    :alt: earth.py

.. |brownian2D.py| replace:: brownian2D.py
.. _brownian2D.py: https://github.com/marcomusy/vedo/tree/master/examples/simulations/brownian2D.py
.. |brownian2D| image:: https://user-images.githubusercontent.com/32848391/50738948-73ce8300-11d9-11e9-8ef6-fc4f64c4a9ce.gif
    :width: 200 px
    :target: brownian2D.py_

.. |gas.py| replace:: gas.py
.. _gas.py: https://github.com/marcomusy/vedo/tree/master/examples/simulations/gas.py
.. |gas| image:: https://user-images.githubusercontent.com/32848391/50738954-7e891800-11d9-11e9-95aa-67c92ca6476b.gif
    :width: 200 px
    :target: gas.py_
    :alt: gas.py

.. |tube.py| replace:: tube.py
.. _tube.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/tube.py
.. |tube| image:: https://user-images.githubusercontent.com/32848391/51801626-adc30000-2240-11e9-8866-9d9d5d8790ab.png
    :width: 200 px
    :target: tube.py_
    :alt: tube.py

.. |mesh_threshold.py| replace:: mesh_threshold.py
.. _mesh_threshold.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_threshold.py
.. |mesh_threshold| image:: https://user-images.githubusercontent.com/32848391/51807663-4762cf80-228a-11e9-9d0c-184bb11a97bf.png
    :width: 200 px
    :target: mesh_threshold.py_
    :alt: mesh_threshold.py

.. |cutWithMesh1.py| replace:: cutWithMesh1.py
.. _cutWithMesh1.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/cutWithMesh1.py
.. |cutWithMesh1| image:: https://user-images.githubusercontent.com/32848391/51808907-e7c0f000-229a-11e9-98a7-fefc7261b3c3.png
    :width: 200 px
    :target: cutWithMesh1.py_
    :alt: cutWithMesh1.py

.. |paraboloid| image:: https://user-images.githubusercontent.com/32848391/51211547-260ef480-1916-11e9-95f6-4a677e37e355.png
    :width: 200 px
    :alt: paraboloid

.. |isosurfaces.py| replace:: isosurfaces.py
.. _isosurfaces.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/isosurfaces.py
.. |isosurfaces| image:: https://user-images.githubusercontent.com/32848391/51558920-ec436e00-1e80-11e9-9d96-aa9b7c72d58b.png
    :width: 200 px
    :target: isosurfaces.py_
    :alt: isosurfaces.py

.. |meshquality.py| replace:: meshquality.py
.. _meshquality.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/meshquality.py
.. |meshquality| image:: https://user-images.githubusercontent.com/32848391/51831269-fb4b7580-22f1-11e9-81ea-13467a5649ca.png
    :width: 200 px
    :target: meshquality.py_
    :alt: meshquality.py

.. |geodesic.py| replace:: geodesic.py
.. _geodesic.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/geodesic.py
.. |geodesic| image:: https://user-images.githubusercontent.com/32848391/51855637-015f4780-232e-11e9-92ca-053a558e7f70.png
    :width: 200 px
    :target: geodesic.py_
    :alt: geodesic.py


.. |cutAndCap.py| replace:: cutAndCap.py
.. _cutAndCap.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/cutAndCap.py
.. |cutAndCap| image:: https://user-images.githubusercontent.com/32848391/51930515-16ee7300-23fb-11e9-91af-2b6b3d626246.png
    :width: 200 px
    :target: cutAndCap.py_
    :alt: cutAndCap.py


.. |convexHull.py| replace:: convexHull.py
.. _convexHull.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/convexHull.py
.. |convexHull| image:: https://user-images.githubusercontent.com/32848391/51932732-068cc700-2400-11e9-9b68-30294a4fa4e3.png
    :width: 200 px
    :target: convexHull.py_
    :alt: convexHull.py

.. |curvature| image:: https://user-images.githubusercontent.com/32848391/51934810-c2e88c00-2404-11e9-8e7e-ca0b7984bbb7.png
    :alt: curvature

.. |progbar| image:: https://user-images.githubusercontent.com/32848391/51858823-ed1f4880-2335-11e9-8788-2d102ace2578.png
    :alt: progressbar

.. |multiwindows| image:: https://user-images.githubusercontent.com/32848391/50738853-be9bcb00-11d8-11e9-9c8e-69864ad7c045.jpg
    :alt: multiwindows

.. |annotations.py| replace:: annotations.py
.. _annotations.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/annotations.py

.. |Cone| image:: https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestCone.png
    :width: 200 px

.. |Cylinder| image:: https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestCylinder.png
    :width: 200 px

.. |Disk| image:: https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestDisk.png
    :width: 200 px

.. |OrientedArrow| image:: https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestOrientedArrow.png
    :width: 200 px

.. |Plane| image:: https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestPlane.png
    :width: 200 px

.. |Polygon| image:: https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestRegularPolygonSource.png
    :width: 200 px

.. |Sphere| image:: https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/GeometricObjects/TestSphere.png
    :width: 200 px


.. |embryoslider| image:: https://user-images.githubusercontent.com/32848391/52141624-975ce000-2656-11e9-8d31-2a3c92ab79d6.png
    :width: 200 px

.. |isosurfaces1| image:: https://user-images.githubusercontent.com/32848391/52141625-975ce000-2656-11e9-91fc-291e072fc4c1.png
    :width: 200 px

.. |splitmesh.py| replace:: splitmesh.py
.. _splitmesh.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/splitmesh.py
.. |splitmesh| image:: https://user-images.githubusercontent.com/32848391/52141626-97f57680-2656-11e9-80ea-fcd3571a6422.png
    :width: 200 px
    :target: splitmesh.py_
    :alt: splitmesh.py

.. |projectsphere.py| replace:: projectsphere.py
.. _projectsphere.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/projectsphere.py
.. |projectsphere| image:: https://user-images.githubusercontent.com/32848391/52144163-c9be0b80-265d-11e9-9ce6-d6f2b919c214.png
    :width: 200 px
    :target: projectsphere.py_
    :alt: projectsphere.py


.. |mesh2volume.py| replace:: mesh2volume.py
.. _mesh2volume.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/mesh2volume.py
.. |mesh2volume| image:: https://user-images.githubusercontent.com/32848391/52168902-5638fe80-2730-11e9-8033-8e470a3d4f0f.jpg
    :width: 200 px
    :target: mesh2volume.py_
    :alt: mesh2volume.py

.. |markpoint.py| replace:: markpoint.py
.. _markpoint.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/markpoint.py
.. |markpoint| image:: https://user-images.githubusercontent.com/32848391/52169969-1fb7af80-2741-11e9-937f-5c331d9a1d11.jpg
    :width: 200 px
    :target: markpoint.py_
    :alt: markpoint.py

.. |readVolumeAsIsoSurface.py| replace:: readVolumeAsIsoSurface.py
.. _readVolumeAsIsoSurface.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/readVolumeAsIsoSurface.py

.. |read_volume2.py| replace:: read_volume2.py
.. _read_volume2.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/read_volume2.py
.. |read_volume2| image:: https://user-images.githubusercontent.com/32848391/50739036-6bc31300-11da-11e9-89b3-04a75187f812.jpg
    :width: 200 px
    :target: read_volume2.py_
    :alt: read_volume2.py

.. |glyphs.py| replace:: glyphs.py
.. _glyphs.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/glyphs.py
.. |glyphs| image:: https://user-images.githubusercontent.com/32848391/52233403-47cd1d00-28bf-11e9-86b0-cbceebbde0de.jpg
    :width: 200 px
    :target: glyphs.py_
    :alt: glyphs.py

.. |glyphs_arrows.py| replace:: glyphs_arrows.py
.. _glyphs_arrows.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/glyphs_arrows.py
.. |glyphs_arrows| image:: https://user-images.githubusercontent.com/32848391/55897850-a1a0da80-5bc1-11e9-81e0-004c8f396b43.jpg
    :width: 200 px
    :target: glyphs_arrows.py_
    :alt: glyphs_arrows.py

.. |interpolateField.py| replace:: interpolateField.py
.. _interpolateField.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/interpolateField.py
.. |interpolateField| image:: https://user-images.githubusercontent.com/32848391/52416117-25b6e300-2ae9-11e9-8d86-575b97e543c0.png
    :width: 200 px
    :target: interpolateField.py_
    :alt: interpolateField.py

.. |rotateImage.py| replace:: rotateImage.py
.. _rotateImage.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/rotateImage.py
.. |rotateImage| image:: https://user-images.githubusercontent.com/32848391/52416910-fb662500-2aea-11e9-88ba-9c73128e8666.jpg
    :width: 200 px
    :target: rotateImage.py_
    :alt: rotateImage.py

.. |basicshapes| image:: https://user-images.githubusercontent.com/32848391/50738811-58af4380-11d8-11e9-9bfb-378c27c9d26f.png
    :alt: basicshapes

.. |lines| image:: https://user-images.githubusercontent.com/32848391/52503049-ac9cb600-2be4-11e9-86af-72a538af14ef.png
    :width: 200 px
    :alt: lines

.. |vlogo_large| image:: https://user-images.githubusercontent.com/32848391/52522716-4fa70b80-2c89-11e9-92a7-0d22cbe34758.png
    :alt: vlogo_large

.. |vlogo_medium| image:: https://user-images.githubusercontent.com/32848391/52522717-503fa200-2c89-11e9-87ab-67eb44652e24.png
    :alt: vlogo_medium

.. |vlogo_small| image:: https://user-images.githubusercontent.com/32848391/52522718-50d83880-2c89-11e9-80ff-df1b5618a84a.png
    :alt: vlogo_small

.. |vlogo_small_dark| image:: https://user-images.githubusercontent.com/32848391/52522719-50d83880-2c89-11e9-8b90-a1c21c27b007.png
    :alt: vlogo_small_dark

.. |vlogo_tube| image:: https://user-images.githubusercontent.com/32848391/52522720-5170cf00-2c89-11e9-8b1d-a7a5cf75e71b.png
    :alt: vlogo_tube

.. |vlogo_tube_dark| image:: https://user-images.githubusercontent.com/32848391/52522721-5170cf00-2c89-11e9-8fbb-6efa13940aa1.png
    :alt: vlogo_tube_dark

.. |fitline.py| replace:: fitline.py
.. _fitline.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/fitline.py
.. |fitline| image:: https://user-images.githubusercontent.com/32848391/50738864-c0658e80-11d8-11e9-8754-c670f1f331d6.jpg
    :width: 200 px
    :target: fitline.py_
    :alt: fitline.py

.. |sliders3d.py| replace:: sliders3d.py
.. _sliders3d.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/sliders3d.py
.. |sliders3d| image:: https://user-images.githubusercontent.com/32848391/52859555-4efcf200-312d-11e9-9290-6988c8295163.png
    :width: 200 px
    :target: sliders3d.py_
    :alt: sliders3d.py

.. |ex01_showmesh.py| replace:: ex01_showmesh.py
.. _ex01_showmesh.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/ex01_show-mesh.py
.. |ex01_showmesh| image:: https://user-images.githubusercontent.com/32848391/53026243-d2d31900-3462-11e9-9dde-518218c241b6.jpg
    :width: 200 px
    :target: ex01_showmesh.py_
    :alt: ex01_showmesh.py

.. |ex02_tetralize-mesh.py| replace:: ex02_tetralize-mesh.py
.. _ex02_tetralize-mesh.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/ex02_tetralize-mesh.py
.. |ex02_tetralize-mesh| image:: https://user-images.githubusercontent.com/32848391/53026244-d2d31900-3462-11e9-835a-1fa9d66d3dae.png
    :width: 200 px
    :target: ex02_tetralize-mesh.py_
    :alt: ex02_tetralize-mesh.py

.. |ex06_elasticity1.py| replace:: ex06_elasticity1.py
.. _ex06_elasticity1.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/ex06_elasticity1.py
.. |ex06_elasticity1| image:: https://user-images.githubusercontent.com/32848391/53026245-d2d31900-3462-11e9-9db4-96211569d114.jpg
    :width: 200 px
    :target: ex06_elasticity1.py_
    :alt: ex06_elasticity1.py

.. |ex06_elasticity2.py| replace:: ex06_elasticity2.py
.. _ex06_elasticity2.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/ex06_elasticity2.py
.. |ex06_elasticity2| image:: https://user-images.githubusercontent.com/32848391/53026246-d36baf80-3462-11e9-96a5-8eaf0bb0f9a4.jpg
    :width: 200 px
    :target: ex06_elasticity2.py_
    :alt: ex06_elasticity2.py


.. |flatarrow.py| replace:: flatarrow.py
.. _flatarrow.py: https://github.com/marcomusy/vedo/tree/master/examples/other/basic/flatarrow.py
.. |flatarrow| image:: https://user-images.githubusercontent.com/32848391/54612632-97c00780-4a59-11e9-8532-940c25a5dfd8.png
    :width: 200 px
    :target: flatarrow.py_
    :alt: flatarrow.py

.. |printhisto| image:: https://user-images.githubusercontent.com/32848391/55073046-03732780-508d-11e9-9bf9-c5de8631dd73.png
    :width: 200 px

.. |distance2mesh.py| replace:: distance2mesh.py
.. _distance2mesh.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/distance2mesh.py
.. |distance2mesh| image:: https://user-images.githubusercontent.com/32848391/55965881-b5a71380-5c77-11e9-8680-5bddceab813a.png
    :width: 200 px
    :target: distance2mesh.py_
    :alt: distance2mesh.py

.. |pendulum.py| replace:: pendulum.py
.. _pendulum.py: https://github.com/marcomusy/vedo/tree/master/examples/simulations/pendulum.py
.. |pendulum| image:: https://user-images.githubusercontent.com/32848391/55420020-51e56200-5576-11e9-8513-4a5d93913b17.png
    :width: 200 px
    :target: pendulum.py_
    :alt: pendulum.py

.. |latex.py| replace:: latex.py
.. _latex.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/latex.py
.. |latex| image:: https://user-images.githubusercontent.com/32848391/55568648-6190b200-5700-11e9-9547-0798c588a7a5.png
    :width: 200 px
    :target: latex.py_
    :alt: latex.py

.. |ft04_heat_gaussian.py| replace:: ft04_heat_gaussian.py
.. _ft04_heat_gaussian.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/ft04_heat_gaussian.py
.. |ft04_heat_gaussian| image:: https://user-images.githubusercontent.com/32848391/55578167-88a5ae80-5715-11e9-84ea-bdab54099887.gif
    :width: 200 px
    :target: ft04_heat_gaussian.py_
    :alt: ft04_heat_gaussian.py

.. |cutcube| image:: https://user-images.githubusercontent.com/32848391/55965516-08cc9680-5c77-11e9-8d23-720f6c088ea2.png
    :width: 200 px

.. |intline| image:: https://user-images.githubusercontent.com/32848391/55967065-eee08300-5c79-11e9-8933-265e1bab9f7e.png
    :width: 200 px

.. |cropped| image:: https://user-images.githubusercontent.com/32848391/57081955-0ef1e800-6cf6-11e9-99de-b45220939bc9.png
    :width: 200 px

.. |dolfinmesh| image:: https://user-images.githubusercontent.com/32848391/53026243-d2d31900-3462-11e9-9dde-518218c241b6.jpg
    :width: 200 px

.. |turing_pattern.py| replace:: turing_pattern.py
.. _turing_pattern.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/turing_pattern.py
.. |turing_pattern| image:: https://user-images.githubusercontent.com/32848391/56056437-77cfeb00-5d5c-11e9-9887-828e5745d547.gif
    :width: 200 px
    :target: turing_pattern.py_
    :alt: turing_pattern.py

.. |demo_cahn-hilliard.py| replace:: demo_cahn-hilliard.py
.. _demo_cahn-hilliard.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/demo_cahn-hilliard.py
.. |demo_cahn-hilliard| image:: https://user-images.githubusercontent.com/32848391/56664730-edb34b00-66a8-11e9-9bf3-73431f2a98ac.gif
    :width: 200 px
    :target: demo_cahn-hilliard.py_
    :alt: demo_cahn-hilliard.py


.. |navier-stokes_lshape.py| replace:: navier-stokes_lshape.py
.. _navier-stokes_lshape.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/navier-stokes_lshape.py
.. |navier-stokes_lshape| image:: https://user-images.githubusercontent.com/32848391/56671156-6bc91f00-66b4-11e9-8c58-e6b71e2ad1d0.gif
    :width: 200 px
    :target: navier-stokes_lshape.py_
    :alt: navier-stokes_lshape.py


.. |mesh_map2cell.py| replace:: mesh_map2cell.py
.. _mesh_map2cell.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_map2cell.py
.. |mesh_map2cell| image:: https://user-images.githubusercontent.com/32848391/56600859-0153a880-65fa-11e9-88be-34fd96b18e9a.png
    :width: 200 px
    :target: mesh_map2cell.py_
    :alt: mesh_map2cell.py


.. |ex03_poisson.py| replace:: ex03_poisson.py
.. _ex03_poisson.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/ex03_poisson.py
.. |ex03_poisson| image:: https://user-images.githubusercontent.com/32848391/54925524-bec18200-4f0e-11e9-9eab-29fd61ef3b8e.png
    :width: 200 px
    :target: ex03_poisson.py_
    :alt: ex03_poisson.py

.. |elastodynamics.py| replace:: elastodynamics.py
.. _elastodynamics.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/elastodynamics.py
.. |elastodynamics| image:: https://user-images.githubusercontent.com/32848391/54932788-bd4a8680-4f1b-11e9-9326-33645171a45e.gif
    :width: 200 px
    :target: elastodynamics.py_
    :alt: elastodynamics.py

.. |ft02_poisson_membrane.py| replace:: ft02_poisson_membrane.py
.. _ft02_poisson_membrane.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/ft02_poisson_membrane.py
.. |ft02_poisson_membrane| image:: https://user-images.githubusercontent.com/32848391/55499287-ed91d380-5645-11e9-8e9a-e31e2e3b1649.jpg
    :width: 200 px
    :target: ft02_poisson_membrane.py_
    :alt: ft02_poisson_membrane.py


.. |stokes.py| replace:: stokes.py
.. _stokes.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/stokes.py
.. |stokes| image:: https://user-images.githubusercontent.com/32848391/55098209-aba0e480-50bd-11e9-8842-42d3f0b2d9c8.png
    :width: 200 px
    :target: stokes.py_
    :alt: stokes.py

.. |stokes1.py| replace:: stokes1.py
.. _stokes1.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/stokes.py
.. |stokes1| image:: https://user-images.githubusercontent.com/32848391/56986911-6116fa00-6b8c-11e9-83f5-5b4efe430c0c.jpg
    :width: 200 px
    :target: stokes1.py_
    :alt: stokes1.py

.. |demo_submesh.py| replace:: demo_submesh.py
.. _demo_submesh.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/demo_submesh.py
.. |demo_submesh| image:: https://user-images.githubusercontent.com/32848391/56675428-4e984e80-66bc-11e9-90b0-43dde7e4cc29.png
    :width: 200 px
    :target: demo_submesh.py_
    :alt: demo_submesh.py

.. |pi_estimate.py| replace:: pi_estimate.py
.. _pi_estimate.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/pi_estimate.py
.. |pi_estimate| image:: https://user-images.githubusercontent.com/32848391/56675429-4e984e80-66bc-11e9-9217-a0652a8e74fe.png
    :width: 200 px
    :target: pi_estimate.py_
    :alt: pi_estimate.py

.. |isolines.py| replace:: isolines.py
.. _isolines.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/isolines.py
.. |isolines| image:: https://user-images.githubusercontent.com/32848391/72433087-f00a8780-3798-11ea-9778-991f0abeca70.png
    :width: 200 px
    :target: isolines.py_
    :alt: isolines.py

.. |inset.py| replace:: inset.py
.. _inset.py: https://github.com/marcomusy/vedo/tree/master/examples/other/inset.py
.. |inset| image:: https://user-images.githubusercontent.com/32848391/56758560-3c3f1300-6797-11e9-9b33-49f5a4876039.jpg
    :width: 200 px
    :target: inset.py_
    :alt: inset.py

.. |legosurface.py| replace:: legosurface.py
.. _legosurface.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/legosurface.py
.. |legosurface| image:: https://user-images.githubusercontent.com/32848391/56820682-da40e500-684c-11e9-8ea3-91cbcba24b3a.png
    :width: 200 px
    :target: legosurface.py_
    :alt: legosurface.py


.. |streamribbons.py| replace:: streamribbons.py
.. _streamribbons.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/streamribbons.py
.. |streamribbons| image:: https://user-images.githubusercontent.com/32848391/56963999-9145a500-6b5a-11e9-9461-0037c471faab.png
    :width: 200 px
    :target: streamribbons.py_
    :alt: streamribbons.py


.. |streamlines1.py| replace:: streamlines1.py
.. _streamlines1.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/streamlines1.py
.. |streamlines1| image:: https://user-images.githubusercontent.com/32848391/56964002-9145a500-6b5a-11e9-9e3f-da712609d896.png
    :width: 200 px
    :target: streamlines1.py_
    :alt: streamlines1.py

.. |streamlines2.py| replace:: streamlines2.py
.. _streamlines2.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/streamlines2.py
.. |streamlines2| image:: https://user-images.githubusercontent.com/32848391/56964001-9145a500-6b5a-11e9-935b-1b2425bd7dd2.png
    :width: 200 px
    :target: streamlines2.py_
    :alt: streamlines2.py

.. |office.py| replace:: office.py
.. _office.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/office.py
.. |office| image:: https://user-images.githubusercontent.com/32848391/56964003-9145a500-6b5a-11e9-9d9e-9736d90e1900.png
    :width: 200 px
    :target: office.py_
    :alt: office.py

.. |value-iteration.py| replace:: value-iteration.py
.. _value-iteration.py: https://github.com/marcomusy/vedo/tree/master/examples/other/value-iteration.py
.. |value-iteration| image:: https://user-images.githubusercontent.com/32848391/56964055-afaba080-6b5a-11e9-99cf-3fac99df9878.jpg
    :width: 200 px
    :target: value-iteration.py_
    :alt: value-iteration.py

.. |magnetostatics.py| replace:: magnetostatics.py
.. _magnetostatics.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/magnetostatics.py
.. |magnetostatics| image:: https://user-images.githubusercontent.com/32848391/56985162-fb287380-6b87-11e9-9cf9-045bd08c3b9b.jpg
    :width: 200 px
    :target: magnetostatics.py_
    :alt: magnetostatics.py

.. |export_x3d.py| replace:: export_x3d.py
.. _export_x3d.py: https://github.com/marcomusy/vedo/tree/master/examples/other/export_x3d.py
.. |export_x3d| image:: https://user-images.githubusercontent.com/32848391/57160341-c6ffbd80-6de8-11e9-95ff-7215ce642bc5.jpg
    :width: 200 px
    :target: export_x3d.py_
    :alt: export_x3d.py


.. |silhouette.py| replace:: silhouette.py
.. _silhouette.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/silhouette1.py
.. |silhouette| image:: https://user-images.githubusercontent.com/32848391/57179369-8e5df380-6e7d-11e9-99b4-3b1a120dd375.png
    :width: 200 px
    :target: silhouette.py_
    :alt: silhouette.py

.. |shadow.py| replace:: shadow.py
.. _shadow.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/shadow.py
.. |shadow| image:: https://user-images.githubusercontent.com/32848391/57312574-1d714280-70ee-11e9-8741-04fc5386d692.png
    :width: 200 px
    :target: shadow.py_
    :alt: shadow.py

.. |airplanes.py| replace:: airplanes.py
.. _airplanes.py: https://github.com/marcomusy/vedo/tree/master/examples/simulations/airplanes.py
.. |airplanes| image:: https://user-images.githubusercontent.com/32848391/57341963-b8910900-713c-11e9-898a-84b6d3712bce.gif
    :width: 200 px
    :target: airplanes.py_
    :alt: airplanes.py


.. |heatconv.py| replace:: heatconv.py
.. _heatconv.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/heatconv.py
.. |heatconv| image:: https://user-images.githubusercontent.com/32848391/57455107-b200af80-726a-11e9-897d-9c7bcb9854ac.gif
    :width: 200 px
    :target: heatconv.py_
    :alt: heatconv.py

.. |scalemesh.py| replace:: scalemesh.py
.. _scalemesh.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/scalemesh.py
.. |scalemesh| image:: https://user-images.githubusercontent.com/32848391/57393382-431c4b80-71c3-11e9-9a2c-8abb172f5468.png
    :width: 200 px
    :target: scalemesh.py_
    :alt: scalemesh.py

.. |elasticbeam.py| replace:: elasticbeam.py
.. _elasticbeam.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/elasticbeam.py
.. |elasticbeam| image:: https://user-images.githubusercontent.com/32848391/57476429-d7a3ae00-7296-11e9-9f50-8f456823ef3d.png
    :width: 200 px
    :target: elasticbeam.py_
    :alt: elasticbeam.py

.. |specular.py| replace:: specular.py
.. _specular.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/specular.py
.. |specular| image:: https://user-images.githubusercontent.com/32848391/57543051-8c030a00-7353-11e9-84cd-b01f3449d255.jpg
    :width: 200 px
    :target: specular.py_
    :alt: specular.py

.. |wavy_1d.py| replace:: wavy_1d.py
.. _wavy_1d.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/wavy_1d.py
.. |wavy_1d| image:: https://user-images.githubusercontent.com/32848391/57570898-70593b80-7407-11e9-87cf-ce498f499c09.gif
    :width: 200 px
    :target: wavy_1d.py_
    :alt: wavy_1d.py

.. |idealpass.link| replace:: idealpass.link
.. _idealpass.link: https://lorensen.github.io/VTKExamples/site/Cxx/ImageProcessing/IdealHighPass
.. |idealpass| image:: https://raw.githubusercontent.com/lorensen/VTKExamples/master/src/Testing/Baseline/Cxx/ImageProcessing/TestIdealHighPass.png
    :width: 200 px
    :target: idealpass.link_

.. |buildmesh.py| replace:: buildmesh.py
.. _buildmesh.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/buildmesh.py
.. |buildmesh| image:: https://user-images.githubusercontent.com/32848391/57858625-b0e2fb80-77f1-11e9-94f0-1973ed86ae70.png
    :width: 200 px
    :target: buildmesh.py_
    :alt: buildmesh.py

.. |customAxes1.py| replace:: customAxes1.py
.. _customAxes1.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/customAxes1.py
.. |customAxes1| image:: https://user-images.githubusercontent.com/32848391/58181826-c605d180-7cac-11e9-9786-11b5eb278f20.png
    :width: 200 px
    :target: customAxes1.py_
    :alt: customAxes1.py

.. |customAxes2.py| replace:: customAxes2.py
.. _customAxes2.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/customAxes2.py

.. |customAxes3.py| replace:: customAxes3.py
.. _customAxes3.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/customAxes3.py

.. |awefem.py| replace:: awefem.py
.. _awefem.py: https://github.com/marcomusy/vedo/tree/master/examples/other/dolfin/awefem.py
.. |awefem| image:: https://user-images.githubusercontent.com/32848391/58368591-8b3fab80-7eef-11e9-882f-8b8eaef43567.gif
    :width: 200 px
    :target: awefem.py_
    :alt: awefem.py

.. |fenics_logo| image:: https://user-images.githubusercontent.com/32848391/58764910-3940fa80-856d-11e9-8160-af89a5ab5d02.gif

.. |warp3.py| replace:: warp3.py
.. _warp3.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/warp3.py
.. |warp3| image:: https://user-images.githubusercontent.com/32848391/59032715-385ae200-8867-11e9-9b07-7f4f8fbfa5bd.png
    :width: 200 px
    :target: warp3.py_
    :alt: warp3.py

.. |interpolateVolume.py| replace:: interpolateVolume.py
.. _interpolateVolume.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/interpolateVolume.py
.. |interpolateVolume| image:: https://user-images.githubusercontent.com/32848391/59095175-1ec5a300-8918-11e9-8bc0-fd35c8981e2b.jpg
    :width: 200 px
    :target: interpolateVolume.py_
    :alt: interpolateVolume.py

.. |deleteMeshPoints.py| replace:: deleteMeshPoints.py
.. _deleteMeshPoints.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/deleteMeshPoints.py
.. |deleteMeshPoints| image:: https://user-images.githubusercontent.com/32848391/59768041-e9b83980-92a3-11e9-94f4-ca1f92540e9f.png
    :width: 200 px
    :target: deleteMeshPoints.py_
    :alt: deleteMeshPoints.py

.. |gray_scott.ipynb| replace:: gray_scott.ipynb
.. _gray_scott.ipynb: https://github.com/marcomusy/vedo/tree/master/examples/simulations/gray_scott.ipynb
.. |gray_scott| image:: https://user-images.githubusercontent.com/32848391/59788744-aaeaa980-92cc-11e9-825d-58da26ca21ff.gif
    :width: 200 px
    :target: gray_scott.ipynb_
    :alt: gray_scott.ipynb

.. |volterra.py| replace:: volterra.py
.. _volterra.py: https://github.com/marcomusy/vedo/tree/master/examples/simulations/volterra.py
.. |volterra| image:: https://user-images.githubusercontent.com/32848391/59788745-aaeaa980-92cc-11e9-93d5-f6a577ba5e4d.png
    :width: 200 px
    :target: volterra.py_
    :alt: volterra.py

.. |tensors.py| replace:: tensors.py
.. _tensors.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/tensors.py
.. |tensors| image:: https://user-images.githubusercontent.com/32848391/59944747-e2d92480-9465-11e9-8012-1fc34a2e30c6.png
    :width: 200 px
    :target: tensors.py_
    :alt: tensors.py

.. |tensor_grid.py| replace:: tensor_grid.py
.. _tensor_grid.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/tensor_grid.py

.. |scalarbars.py| replace:: scalarbars.py
.. _scalarbars.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/scalarbars.py
.. |scalarbars| image:: https://user-images.githubusercontent.com/32848391/62940174-4bdc7900-bdd3-11e9-9713-e4f3e2fdab63.png
    :width: 200 px
    :target: scalarbars.py_
    :alt: scalarbars.py

.. |erode_dilate.py| replace:: erode_dilate.py
.. _erode_dilate.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/erode_dilate.py

.. |vol2points.py| replace:: vol2points.py
.. _vol2points.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/vol2points.py

.. |euclDist.py| replace:: euclDist.py
.. _euclDist.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/euclDist.py

.. |volumeFromMesh.py| replace:: volumeFromMesh.py
.. _volumeFromMesh.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/volumeFromMesh.py

.. |numpy2volume1.py| replace:: numpy2volume1.py
.. _numpy2volume1.py: https://github.com/marcomusy/vedo/tree/master/examples/volumetric/numpy2volume1.py

.. |G_Of_Omega| image:: https://wikimedia.org/api/rest_v1/media/math/render/svg/9c4d02a66b6ff279aae0c4bf07c25e5727d192e4

.. |wikiphong| image:: https://upload.wikimedia.org/wikipedia/commons/6/6b/Phong_components_version_4.png

.. |animation1.py| replace:: animation1.py
.. _animation1.py: https://github.com/marcomusy/vedo/tree/master/examples/other/animation1.py
.. |animation1| image:: https://user-images.githubusercontent.com/32848391/64273764-4b528080-cf42-11e9-90aa-2d88df239871.gif
    :width: 200 px
    :target: animation1.py_
    :alt: animation1.py

.. |animation2.py| replace:: animation2.py
.. _animation2.py: https://github.com/marcomusy/vedo/tree/master/examples/other/animation2.py
.. |animation2| image:: https://user-images.githubusercontent.com/32848391/64273191-1a258080-cf41-11e9-8a18-f192f05f11a9.gif
    :width: 200 px
    :target: animation2.py_
    :alt: animation2.py

.. |polarHisto.py| replace:: polarHisto.py
.. _polarHisto.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/polarHisto.py
.. |polarHisto| image:: https://user-images.githubusercontent.com/32848391/64912717-5754f400-d733-11e9-8a1f-612165955f23.png
    :width: 200 px
    :target: polarHisto.py_
    :alt: polarHisto.py

.. |histo_polar.py| replace:: histo_polar.py
.. _histo_polar.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_polar.py
.. |histo_polar| image:: https://user-images.githubusercontent.com/32848391/64992590-7fc82400-d8d4-11e9-9c10-795f4756a73f.png
    :width: 200 px
    :target: histo_polar.py_
    :alt: histo_polar.py

.. |donut.py| replace:: donut.py
.. _donut.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/donut.py
.. |donut| image:: https://user-images.githubusercontent.com/32848391/64998178-6f6b7580-d8e3-11e9-9bd8-8dfb9ccd90e4.png
    :width: 200 px
    :target: donut.py_
    :alt: donut.py

.. |extrude.py| replace:: extrude.py
.. _extrude.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/extrude.py
.. |extrude| image:: https://user-images.githubusercontent.com/32848391/65963682-971e1a00-e45b-11e9-9f29-05522ae4a800.png
    :width: 200 px
    :target: extrude.py_
    :alt: extrude.py

.. |kspline| image:: https://user-images.githubusercontent.com/32848391/65975805-73fd6580-e46f-11e9-8957-75eddb28fa72.png
    :width: 200 px

.. |mesh_lut.py| replace:: mesh_lut.py
.. _mesh_lut.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/mesh_lut.py

.. |elevation| image:: https://user-images.githubusercontent.com/32848391/68478872-3986a580-0231-11ea-8245-b68a683aa295.png
    :width: 200 px

.. |paramshapes| image:: https://user-images.githubusercontent.com/32848391/69181075-bb6aae80-0b0e-11ea-92f7-d0cd3b9087bf.png
    :width: 400 px

.. |warpto| image:: https://user-images.githubusercontent.com/32848391/69259878-3c817e80-0bbf-11ea-9025-03b9f6affccc.png
    :width: 200 px

.. |linInterpolate.py| replace:: linInterpolate.py
.. _linInterpolate.py: https://github.com/marcomusy/vedo/tree/master/examples/basic/linInterpolate.py
.. |linInterpolate| image:: https://user-images.githubusercontent.com/32848391/70559826-a621f680-1b87-11ea-89f3-e6b74d8953d9.png
    :width: 200 px
    :target: linInterpolate.py_
    :alt: linInterpolate.py

.. |plot_errbars.py| replace:: plot_errbars.py
.. _plot_errbars.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_errbars.py
.. |plot_errbars| image:: https://user-images.githubusercontent.com/32848391/69158509-d6c1c380-0ae6-11ea-9dbf-ff5cd396a9a6.png
    :width: 200 px
    :target: plot_errbars.py_
    :alt: plot_errbars.py

.. |quiver.py| replace:: quiver.py
.. _quiver.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/quiver.py
.. |quiver| image::  https://user-images.githubusercontent.com/32848391/72261438-199aa600-3615-11ea-870e-e44ca4c4b8d3.png
    :width: 200 px
    :target: quiver.py_
    :alt: quiver.py

.. |plot_spheric.py| replace:: plot_spheric.py
.. _plot_spheric.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_spheric.py
.. |plot_spheric| image:: https://user-images.githubusercontent.com/32848391/72433091-f0a31e00-3798-11ea-86bd-6c522e23ec61.png
    :width: 200 px
    :target: plot_spheric.py_
    :alt: plot_spheric.py

.. |fcomplex| image:: https://user-images.githubusercontent.com/32848391/73392962-1709a300-42db-11ea-9278-30c9d6e5eeaa.png
    :width: 200 px

.. |histo_spheric.py| replace:: histo_spheric.py
.. _histo_spheric.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_spheric.py
.. |histo_spheric| image:: https://user-images.githubusercontent.com/32848391/73392901-fccfc500-42da-11ea-828a-9bad6982a823.png
    :width: 200 px
    :target: histo_spheric.py_
    :alt: histo_spheric.py

.. |sphericgrid| image:: https://user-images.githubusercontent.com/32848391/72433092-f0a31e00-3798-11ea-85f7-b2f5fcc31568.png
    :width: 200 px

.. |histo_2D.py| replace:: histo_2D.py
.. _histo_2D.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/histo_2D.py
.. |histo_2D| image:: https://user-images.githubusercontent.com/32848391/74361190-c019c880-4dc6-11ea-9c72-0f2a890e6664.png
    :width: 200 px
    :target: histo_2D.py_
    :alt: histo_2D.py

.. |plot_errband.py| replace:: plot_errband.py
.. _plot_errband.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_errband.py
.. |plot_errband| image:: https://user-images.githubusercontent.com/32848391/73483464-c019d180-439f-11ea-9a8c-59fa49e9ecf4.png
    :width: 200 px
    :target: plot_errband.py_
    :alt: plot_errband.py

.. |plot_pip.py| replace:: plot_pip.py
.. _plot_pip.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_pip.py
.. |plot_pip| image:: https://user-images.githubusercontent.com/32848391/73393632-4ff64780-42dc-11ea-8798-45a81c067f45.png
    :width: 200 px
    :target: plot_pip.py_
    :alt: plot_pip.py

.. |scatter1.py| replace:: scatter1.py
.. _scatter1.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/scatter1.py
.. |scatter1| image:: https://user-images.githubusercontent.com/32848391/72615028-013bcb80-3934-11ea-8ab8-823f1916bc6c.png
    :width: 200 px
    :target: scatter1.py_
    :alt: scatter1.py

.. |scatter2.py| replace:: scatter2.py
.. _scatter2.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/scatter2.py
.. |scatter2| image:: https://user-images.githubusercontent.com/32848391/72446102-2d7c0e80-37b3-11ea-8fe4-b27526af574f.png
    :width: 200 px
    :target: scatter2.py_
    :alt: scatter2.py

.. |scatter3.py| replace:: scatter3.py
.. _scatter3.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/scatter3.py
.. |scatter3| image:: https://user-images.githubusercontent.com/32848391/72446102-2d7c0e80-37b3-11ea-8fe4-b27526af574f.png
    :width: 200 px
    :target: scatter3.py_
    :alt: scatter3.py

.. |customIndividualAxes.py| replace:: customIndividualAxes.py
.. _customIndividualAxes.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/customIndividualAxes.py
.. |customIndividualAxes| image:: https://user-images.githubusercontent.com/32848391/72752870-ab7d5280-3bc3-11ea-8911-9ace00211e23.png
    :width: 200 px
    :target: customIndividualAxes.py_
    :alt: customIndividualAxes.py

.. |plot_stream.py| replace:: plot_stream.py
.. _plot_stream.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_stream.py
.. |plot_stream| image:: https://user-images.githubusercontent.com/32848391/73614123-93162a80-45fc-11ea-969b-9a3293b26f35.png
    :width: 250 px
    :target: plot_stream.py_
    :alt: plot_stream.py

.. |simpleplot| image:: https://user-images.githubusercontent.com/32848391/74363882-c3638300-4dcb-11ea-8a78-eb492ad9711f.png
    :width: 200 px

.. |warpv| image:: https://user-images.githubusercontent.com/32848391/77864546-7a577900-7229-11ea-84ce-4e8e6eeff27f.png
    :width: 200 px

.. |lineage_graph.py| replace:: lineage_graph.py
.. _lineage_graph.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/graph_lineage.py
.. |lineage_graph| image:: https://user-images.githubusercontent.com/32848391/80291851-8152a800-8751-11ea-893e-4a0bb85397b1.png
    :width: 200 px
    :target: lineage_graph.py_
    :alt: lineage_graph.py

.. |bezier| image:: https://user-images.githubusercontent.com/32848391/90437534-dafd2a80-e0d2-11ea-9b93-9ecb3f48a3ff.png
    :width: 200 px

.. |goniometer.py| replace:: goniometer.py
.. _goniometer.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/goniometer.py
.. |goniometer| image:: https://user-images.githubusercontent.com/32848391/90437544-dd5f8480-e0d2-11ea-8321-b52d073444c4.png
    :width: 200 px
    :target: goniometer.py_
    :alt: goniometer.py

.. |intersect2d.py| replace:: intersect2d.py
.. _intersect2d.py: https://github.com/marcomusy/vedo/tree/master/examples/advanced/intersect2d.py
.. |intersect2d| image:: https://user-images.githubusercontent.com/32848391/90437548-de90b180-e0d2-11ea-8e0c-d821db4da8a9.png
    :width: 200 px
    :target: intersect2d.py_
    :alt: intersect2d.py

.. |fonts.py| replace:: fonts.py
.. _fonts.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/fonts3d.py
.. |fonts| image:: https://user-images.githubusercontent.com/32848391/90437539-dcc6ee00-e0d2-11ea-8381-93d211b1bc85.png
    :width: 400 px
    :target: fonts.py_
    :alt: fonts.py

.. |graph_network.py| replace:: graph_network.py
.. _graph_network.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/graph_network.py
.. |graph_network| image:: https://user-images.githubusercontent.com/32848391/90437546-ddf81b00-e0d2-11ea-84d5-e4356a5c5f85.png
    :width: 200 px
    :target: graph_network.py_
    :alt: graph_network.py

.. |plot_density3d.py| replace:: plot_density3d.py
.. _plot_density3d.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/plot_density3d.py
.. |plot_density3d| image:: https://user-images.githubusercontent.com/32848391/90437537-dc2e5780-e0d2-11ea-982c-8dafd467c3cd.png
    :width: 200 px
    :target: plot_density3d.py_
    :alt: plot_density3d.py

.. |fonts3d| image:: https://user-images.githubusercontent.com/32848391/90437540-dd5f8480-e0d2-11ea-8ddc-8839688979d0.png
    :width: 200 px

.. |fontlist| image:: https://user-images.githubusercontent.com/32848391/90437539-dcc6ee00-e0d2-11ea-8381-93d211b1bc85.png
    :width: 200 px

.. |caption.py| replace:: captions.py
.. _caption.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/caption.py
.. |caption| image:: https://user-images.githubusercontent.com/32848391/90437536-dc2e5780-e0d2-11ea-8951-f905ffb54f54.png
    :width: 200 px
    :target: caption.py_
    :alt: caption.py

.. |flag_labels.py| replace:: flag_labels.py
.. _flag_labels.py: https://github.com/marcomusy/vedo/tree/master/examples/other/flag_labels.py
.. |flag_labels| image:: https://user-images.githubusercontent.com/32848391/90620799-3b938100-e213-11ea-80b1-e05ce2949d3a.png
    :width: 200 px
    :target: flag_labels.py_
    :alt: flag_labels.py

.. |whiskers.py| replace:: whiskers.py
.. _whiskers.py: https://github.com/marcomusy/vedo/tree/master/examples/pyplot/whiskers.py
.. |whiskers| image:: https://user-images.githubusercontent.com/32848391/95772479-170cd000-0cbd-11eb-98c4-20c5ca342cb8.png
    :width: 200 px
    :target: whiskers.py_
    :alt: whiskers.py
"""

