from vedo.pyplot import np, histogram
data = np.random.randn(1000)
histogram(data).show().close()
