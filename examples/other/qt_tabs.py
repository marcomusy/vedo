from PyQt5 import QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Mesh, dataurl, Plotter

from qttabsui import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.vtkLayout.addWidget(self.vtkWidget)

        self.plt = Plotter(qtWidget=self.vtkWidget, axes=1)

        self.plt += Mesh(dataurl+'shark.ply').c('cyan')

        self.plt.show()

    def onClose(self):
        print("Disable the interactor before closing to prevent it from trying to act on a already deleted items")
        self.vtkWidget.close()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    app.aboutToQuit.connect(window.onClose)  # <-- connect the onClose event
    window.show()
    sys.exit(app.exec_())
