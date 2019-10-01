from PyQt5 import QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vtkplotter import load, datadir, Plotter

from qt_tabs_ui import Ui_MainWindow


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.vtkWidget = QVTKRenderWindowInteractor(self)
        self.vtkLayout.addWidget(self.vtkWidget)
        
        self.plt = Plotter(qtWidget=self.vtkWidget, axes=1, bg='white')
        
        self.plt += load(datadir+'shark.ply').c('cyan')
        
        self.plt.show()


if __name__ == "__main__":
    import sys

    app = QtWidgets.QApplication(sys.argv)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
