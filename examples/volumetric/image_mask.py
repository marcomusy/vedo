from vedo import Picture, show, settings
from vedo.pyplot import histogram
import numpy as np

settings.defaultFont = "Theemim"

pic = Picture("https://aws1.discourse-cdn.com/business4/uploads/imagej/original/3X/5/8/58468da123203f6056ca786adf159064db47aefa.jpeg")
msh = pic.tomesh()                # convert it to a quad-mesh
rgb = msh.pointdata["RGBA"]       # numpy array

tot = np.sum(rgb, axis=1) + 0.1   # add 0.1 to avoid divide by zero
ratio_g = rgb[:,1] / tot
ratio_r = rgb[:,0] / tot

ids_r = np.where(ratio_r > 0.38)  # threshold to find the red vase
ids_g = np.where(ratio_g > 0.36)  # threshold for grass
ids_w = np.where(tot > 240*3)     # threshold to identify white areas

data_g = np.zeros(msh.N())
data_r = np.zeros(msh.N())
data_w = np.zeros(msh.N())
data_r[ids_r] = 1.0
data_g[ids_g] = 1.0
data_w[ids_w] = 1.0

ngreen = len(ids_g[0])
total  = len(rgb) - len(ids_r[0]) - len(ids_w[0])
gvalue = int(ngreen/total*100 + 0.5)

show([
      [pic, pic.box().lw(3), "Original image. How much grass is there?"],
      histogram(ratio_g, logscale=True, xtitle='ratio of green'),
      [msh.clone().cmap('Greens', data_g), f'Ratio of green is \approx {gvalue}%'],
      [msh.clone().cmap('Reds',   data_r), 'Masking the vase region'],
      [msh.clone().cmap('Greys',  data_w), 'Masking bright areas'],
     ],
     shape="2|3", size=(1370, 1130), sharecam=False,
     bg='aliceblue', mode='image', zoom=1.5, interactive=True,
)
