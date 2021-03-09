from vedo import VedoLogo, settings

settings.useParallelProjection = 1
settings.screenshotTransparentBackground = 0

vl = VedoLogo(frame=False, simple=True, c='k')
vl.show(size=(340,125), zoom=2.8).screenshot("logo_vedo_simple.png")