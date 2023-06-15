import sys
# from PySide2 import QtWidgets, QtCore
from PyQt5 import Qt
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, Mesh, BoxCutter, dataurl

class MainWindow(Qt.QMainWindow):

    def __init__(self, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.frame = Qt.QFrame()
        self.layout = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        # Create renderer and add the vedo objects and callbacks
        self.plt = Plotter(qt_widget=self.vtkWidget)
        mesh = Mesh(dataurl+'cow.vtk')
        self.cutter = BoxCutter(mesh)
        self.plt += [mesh, self.cutter]
        self.plt.show()

        box_cutter_button_on = Qt.QPushButton("Start the box cutter")
        box_cutter_button_on.clicked.connect(self.ctool_start)

        box_cutter_button_off = Qt.QPushButton("Stop the box cutter")
        box_cutter_button_off.clicked.connect(self.ctool_stop)

        # Set-up the rest of the Qt window
        self.layout.addWidget(self.vtkWidget)
        self.layout.addWidget(box_cutter_button_on)
        self.layout.addWidget(box_cutter_button_off)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
        self.show()

    def ctool_start(self):
        self.cutter.on()

    def ctool_stop(self):
        self.cutter.off()

    def onClose(self):
        #Disable the interactor before closing to prevent it
        #from trying to act on already deleted items
        self.vtkWidget.close()

if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.onClose) # <-- connect the onClose event
    app.exec_()
