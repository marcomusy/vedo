"""Whisker plot with quantiles indication
(horizontal line shows the mean value)"""
from vedo import np, settings, Axes, Brace, Line, Ribbon, show
from vedo.pyplot import whisker

settings.default_font = "Theemim"

# build some theoretical expectation to be shown as a grey band
x = np.linspace(-1, 9, 100)
y = x/5 + 0.2*np.sin(x)
ye= y**2/5 + 0.1 # error on y
line = Line(np.c_[x, y])
band = Ribbon(np.c_[x, y-ye], np.c_[x, y+ye]).c('black',0.1)

# create 5 whisker bars with some random data
ws = []
for i in range(5):
    xval = i*2 # position along x axis
    data = xval/5 + 0.2*np.sin(xval) + np.random.randn(25)
    w = whisker(data, bc=i, s=0.5).x(xval)
    ws.append(w)
    # print(i, 'whisker:\n', w.info)

# build braces to inndicate stats significance and dosage
bra1 = Brace([0, 3],[2, 3], comment='*~*', s=0.7, style='[')
bra2 = Brace([4,-1],[8,-1], comment='dose > 3~:mug/kg', s=0.4)

# build custom axes
axes = Axes(xrange=[-1,9],
            yrange=[-3,5],
            htitle=':beta_c  expression: change in time',
            xtitle=' ',
            ytitle='Level of :beta_c  protein in \muM/l',
            x_values_and_labels=[(0,'Experiment^A\n at t=1h'),
                                 (4,'Experiment^B\n at t=2h'),
                                 (8,'Experiment^C\n at t=4h'),
                                ],
            xlabel_size=0.02,
            xygrid=False,
           )

show(ws, bra1, bra2, line, band, __doc__, axes, zoom=1.3).close()


