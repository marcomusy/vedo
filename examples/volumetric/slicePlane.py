from vedo import load, datadir, show

vol = load(datadir+'embryo.slc').alpha([0,0,1]).c('hot_r')
#vol.addScalarBar(title='Volume', pos=(0.8,0.55))
vol.addScalarBar3D(title='Voxel intensity', c='k')

sl = vol.slicePlane(origin=vol.center(), normal=(0,1,1))

sl.cmap('viridis').lighting('ambient').addScalarBar(title='Slice')

show(vol, sl, "Slice a Volume with an arbitrary plane",
     axes=1, viewup='z'
)