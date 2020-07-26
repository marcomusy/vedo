"""
A sort of minimal example of how to embed a rendering window
into a qt application.
"""
print(__doc__)
import sys
from PyQt5 import Qt

# You may need to uncomment these lines on some systems:
#import vtk.qt
#vtk.qt.QVTKRWIBase = "QGLWidget"

from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from vedo import Plotter, Cone

class MainWindow(Qt.QMainWindow):
    def __init__(self, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.frame = Qt.QFrame()
        self.vl = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        # create renderer and add the actors
        self.vp = Plotter(qtWidget=self.vtkWidget)

        self.vp += Cone().rotateX(20)
        self.vp.show(interactorStyle=0)

        # set-up the rest of the Qt window
        button = Qt.QPushButton("My Button makes the cone red")
        button.setToolTip('This is an example button')
        button.clicked.connect(self.onClick)
        self.vl.addWidget(button)
        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)
        self.show() # <--- show the Qt Window

    def onClose(self):
        #Disable the interactor before closing to prevent it
        #from trying to act on already deleted items
        self.vtkWidget.close()

    @Qt.pyqtSlot()
    def onClick(self):
        self.vp.actors[0].color('red').rotateZ(40)
        self.vp.interactor.Render()

if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.onClose) # <-- connect the onClose event
    app.exec_()
