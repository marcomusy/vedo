import numpy as np

class Cell:
    
    def __init__(self, name='cell', c='b', pos=[0,0,0]):
        self.vp    = None
        self.name  = name
        self.status = None
        self.type  = 0
        self.links = []            # links to other cells
        self.size  = 0.1
        self.color = c
        self.alpha = 0.025

        self._pos = pos
        self._vel = [0,0,0]
        self._pol = None
        
        self.timeOfBirth = 0       # time of life since birth
        self.tdiv        = 2       # after this time cell will divide on average
        self.tdiv_spread = 0.5     # gaussian spread to determine division time
        self.split_dist  = 0.2       # split at this initial distance 
        self.split_dist_spread = 0.02  # gauss split initial distance spread

        self.apoptosis = 3
        self.apoptosis_spread = 0.5
     
        self.surface = None
        self.nucleus = None
        self.line    = None

        self.showPolarity = False


    def build(self, vp):

        #decide life and death
        self.tdiv      += np.random.randn()*self.tdiv_spread
        self.tdiv       = np.abs(self.tdiv)
        self.apoptosis += np.random.randn()*self.apoptosis_spread
        self.apoptosis  = np.abs(self.apoptosis)
                
        self.vp = vp
        self.surface = vp.sphere(self._pos, r=self.size, 
                                 c=self.color, alpha=self.alpha, res=48)
        self.nucleus = vp.sphere(self._pos, r=self.size/6., 
                                 c=self.color, alpha=1, res=12)

        
    def shouldDivide(self, t):
        if t > self.timeOfBirth + self.tdiv: 
            return True
        else:
            return False

    def shouldDie(self, t):
        if t > self.timeOfBirth + self.apoptosis: 
            return True
        else:
            return False

    def split(self, t):
        c = Cell(self.name+'.')
        phi = np.random.uniform()*np.pi*2.
        costheta = np.random.uniform()*2. - 1
        u = np.abs(np.random.normal()) * self.split_dist_spread
        theta = np.arccos( costheta )
        radius = self.split_dist * u**(1/3.)
        x = radius*np.sin(theta)*np.cos(phi)
        y = radius*np.sin(theta)*np.sin(phi)
        z = radius*np.cos(theta)
        d = np.array([x,y,z])/2.
        c.setPos(self._pos + d)
        self._pos = self._pos - d
        c.setPolarity(self._pol)
        c.setVel(self._vel)
        c.color = self.color
        c.build(self.vp)
        c.surface.SetPosition(c._pos)
        c.nucleus.SetPosition(c._pos)
        self.surface.SetPosition(self._pos)
        self.nucleus.SetPosition(self._pos)
        self.timeOfBirth = t
        c.timeOfBirth = t
        return c
    
    def remove(self):
        self.vp.removeActor(self.surface)
        self.vp.removeActor(self.nucleus)

    def pos(self): return self._pos
    def setPos(self, x): 
        self._pos = x
        if self.surface:
            self.surface.SetPosition(x)
            self.nucleus.SetPosition(x)
    def addPos(self, x): 
        self._pos += x
        self.surface.SetPosition(self._pos)
        self.nucleus.SetPosition(self._pos)

    def vel(self): return self._vel
    def setVel(self, v): self._vel = v

    def polarity(self): return self._pol
    def setPolarity(self, p): self._pol = p
                
    def dist(self, c):
        v = self._pos - np.array(c._pos)
        s = np.linalg.norm(v)
        return v/s, s
    
    
    
    
    