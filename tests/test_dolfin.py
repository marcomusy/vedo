from vtkplotter import Cone
import dolfin

c1 = Cone()
c2 = c1.clone()

assert c1.N() == c2.N()
