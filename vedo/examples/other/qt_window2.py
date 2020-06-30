"""
A sort of minimal example of how to embed a rendering window
into a qt application.
"""
print(__doc__)

import sys
from PyQt5 import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor

from vedo import Plotter, Cube, Torus, Cone


class MainWindow(Qt.QMainWindow):
    def __init__(self, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.frame = Qt.QFrame()
        self.vl = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)
        self.vl.addWidget(self.vtkWidget)

        vp = Plotter(qtWidget=self.vtkWidget, axes=2, N=2)

        cn = Cone()
        cc = Cube().pos(1, 1, 1).color("pink")
        ss = Torus()
        vp.show(cn, cc, at=0)
        vp.show(ss, at=1, viewup="z", interactorStyle=0)

        self.start(vp)

    def start(self, vp):

        for r in vp.renderers:
            self.vtkWidget.GetRenderWindow().AddRenderer(r)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        self.iren.AddObserver("LeftButtonPressEvent", vp._mouseleft)
        self.iren.AddObserver("RightButtonPressEvent", vp._mouseright)
        self.iren.AddObserver("MiddleButtonPressEvent", vp._mousemiddle)

        def keypress(obj, e):
            vp._keypress(obj, e)
            if self.iren.GetKeySym() in ["q", "space"]:
                self.iren.ExitCallback()
                exit()
        self.iren.AddObserver("KeyPressEvent", keypress)

        self.frame.setLayout(self.vl)
        self.setCentralWidget(self.frame)
        self.show()  # qt not Plotter method
        r.ResetCamera()
        self.iren.Start()

    def onClose(self):
        print("Disable the interactor before closing to prevent it from trying to act on a already deleted items")
        self.vtkWidget.close()


if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.onClose)
    app.exec_()
