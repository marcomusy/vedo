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

from vtkplotter import Plotter, Cone

class MainWindow(Qt.QMainWindow):
    def __init__(self, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.frame = Qt.QFrame()
        self.vl = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        vp = Plotter(qtWidget=self.vtkWidget, axes=4, bg='white')

        vp += Cone()
        vp.show()      # create renderer and add the actors

        # set-up the rest of the Qt window
        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.show()    # <--- show the Qt Window

    def onClose(self):
        print("Disable the interactor before closing to prevent it from trying to act on a already deleted items")
        self.vtkWidget.close()


if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.onClose) # <-- connect the onClose event
    app.exec_()
