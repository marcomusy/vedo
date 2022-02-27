"""
Show a cube for each available texture name.
Any jpg file can be used as texture.
"""
from vedo import dataurl, settings, show, Cube

textures_path = dataurl+'textures/'

print(__doc__)
print('example textures:', textures_path)

cubes = []
cubes.append(Cube().texture(textures_path+'leather.jpg'))
cubes.append(Cube().texture(textures_path+'paper2.jpg'))
cubes.append(Cube().texture(textures_path+'wood1.jpg'))
cubes.append(Cube().texture(textures_path+'wood2.jpg'))

show(cubes, N=4, bg2='lightblue').close()
