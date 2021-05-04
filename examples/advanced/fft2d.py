# 2D Fast Fourier Transform of a picture
from vedo import Picture, show

# url = 'https://comps.canstockphoto.com/a-capital-silhouette-stock-illustrations_csp31110154.jpg'
url = 'https://vedo.embl.es/examples/data/images/dog.jpg'

pic = Picture(url).resize([200,None])  # resize so that x has 200 pixels, but keep y aspect-ratio
picfft = pic.fft(logscale=12)
picfft = picfft.tomesh().cmap('Set1',"RGBA").addScalarBar("12\dotlog(fft)")  # optional step

show([
      [pic, f"Original image\n{url[-40:]}"],
      [picfft, "2D Fast Fourier Transform"],
      [pic.fft(mode='complex').rfft(), "Reversed FFT"],
     ], N=3, bg='gray7', axes=1,
).close()


