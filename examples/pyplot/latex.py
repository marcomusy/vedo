from vedo import Latex

# https://matplotlib.org/tutorials/text/mathtext.html

latex1 = r'x= \frac{ - b \pm \sqrt {b^2 - 4ac} }{2a}'
latex2 = r'\mathcal{A}\mathrm{sin}(2 \omega t)'
latex3 = r'I(Y | X)=\sum_{x \in \mathcal{X}, y \in \mathcal{Y}} p(x, y) \log \left(\frac{p(x)}{p(x, y)}\right)'
latex4 = r'\Gamma_{\epsilon}(x)=\left[1-e^{-2 \pi \epsilon}\right]^{1-x} \prod_{n=0}^{\infty} \frac{1-\exp (-2 \pi \epsilon(n+1))}{1-\exp (-2 \pi \epsilon(x+n))}'
latex5 = r'\left( \begin{array}{l}{c t^{\prime}} \\ {x^{\prime}} \\ {y^{\prime}} \\ {z^{\prime}}\end{array}\right)=\left( \begin{array}{cccc}{\gamma} & {-\gamma \beta} & {0} & {0} \\ {-\gamma \beta} & {\gamma} & {0} & {0} \\ {0} & {0} & {1} & {0} \\ {0} & {0} & {0} & {1}\end{array}\right) \left( \begin{array}{l}{c t} \\ {x} \\ {y} \\ {z}\end{array}\right)'
latex6 = r'\mathrm{CO}_{2}+6 \mathrm{H}_{2} \mathrm{O} \rightarrow \mathrm{C}_{6} \mathrm{H}_{12} \mathrm{O}_{6}+6 \mathrm{O}_{2}'
latex7 = r'x \mathrm{(arb. units)}'

ltx = Latex(latex4, s=1, c='darkblue', bg='', alpha=0.9, usetex=False, fromweb=False)
ltx.crop(0.3, 0.3) # crop top and bottom 30%
ltx.pos(2,0,0)

ltx.show(axes=1)
