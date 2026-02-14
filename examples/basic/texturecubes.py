"""
Show a cube for each available texture name.
Any jpg file can be used as texture.
"""
from vedo import dataurl, show, Cube

textures_path = dataurl+'textures/'

print(__doc__)
print('example textures:', textures_path)

# Build one textured cube per image.
cubes = [
    Cube().texture(textures_path + 'leather.jpg'),
    Cube().texture(textures_path + 'paper2.jpg'),
    Cube().texture(textures_path + 'wood1.jpg'),
    Cube().texture(textures_path + 'wood2.jpg'),
]

show(cubes, N=4, bg2='lightblue').close()
