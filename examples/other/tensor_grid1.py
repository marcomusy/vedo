from vedo import Grid, Tensors, show

domain = Grid(res=[5,5])

# Generate random attributes on this mesh
domain.generate_random_data()

ts = Tensors(domain, scale=0.1)
ts.print()

show(domain, ts).close()

