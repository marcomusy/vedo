from vedo import Latex

# https://matplotlib.org/tutorials/text/mathtext.html

latex1 = r'x= \frac{ - b \pm \sqrt {b^2 - 4ac} }{2a}'
latex2 = r'\mathcal{A}\mathrm{sin}(2 \omega t)'
latex3 = r'I(Y | X)=\sum_{x \in \mathcal{X}, y \in \mathcal{Y}} p(x, y) \log \left(\frac{p(x)}{p(x, y)}\right)'
latex4 = r'\Gamma_{\epsilon}(x)=\left[1-e^{-2 \pi \epsilon}\right]^{1-x} \prod_{n=0}^{\infty} \frac{1-\exp (-2 \pi \epsilon(n+1))}{1-\exp (-2 \pi \epsilon(x+n))}'

ltx = Latex(latex4, s=1, c='darkblue', bg='', usetex=False, res=40)
ltx.crop(0.3, 0.3) # crop top and bottom 30%
ltx.pos(2,0,0)

ltx.show(axes=8, size=(1400,700), bg2='lb', zoom='tight').close()
