# 2D Fast Fourier Transform of a image
from vedo import Image, show

url = 'https://vedo.embl.es/examples/data/images/dog.jpg'

img = Image(url).resize([200,None])  # resize so that x has 200 pixels, but keep y aspect-ratio
img_fft = img.fft(logscale=12)
img_fft = img_fft.tomesh().cmap('Set1',"RGBA").add_scalarbar("12\dotlog(fft)")  # optional step

show([
      [img,    f"Original image\n{url[-40:]}"],
      [img_fft, "2D Fast Fourier Transform"],
      [img.fft(mode='complex').rfft(), "Reversed FFT"],
     ],
     N=3, bg='gray7', axes=1,
).close()


