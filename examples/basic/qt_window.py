import sys
from PyQt5 import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from vtkplotter import Plotter, Cone

# settings.usingQt = True <-- not needed anymore, automatically triggered by passing a qtWidget to Plotter


class MainWindow(Qt.QMainWindow):
    def __init__(self, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.frame = Qt.QFrame()
        self.vl = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        vp = Plotter(qtWidget=self.vtkWidget, axes=4, verbose=False)

        vp += Cone()
        vp.show()      # to create renderer and add the actors

        # set-up the rest of the Qt window
        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)

        self.show() # <--- show the Qt Window


if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    app.exec_()