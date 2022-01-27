import sys
from PyQt5 import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, Picture, Text2D, printc

class MainWindow(Qt.QMainWindow):

    def __init__(self, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.frame = Qt.QFrame()
        self.layout = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        # Create vedo renderer and add objects and callbacks
        self.plt = Plotter(qtWidget=self.vtkWidget)
        self.cbid = self.plt.addCallback("key press", self.onKeypress)
        self.imgActor = Picture("https://icatcare.org/app/uploads/2018/07/Helping-your-new-cat-or-kitten-settle-in-1.png")
        self.text2d = Text2D("Use slider to change contrast")

        self.slider = Qt.QSlider(1)
        self.slider.valueChanged.connect(self.onSlider)
        self.layout.addWidget(self.vtkWidget)
        self.layout.addWidget(self.slider)

        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
        self.plt.show(self.imgActor, self.text2d, mode='image') # build the vedo rendering
        self.show()                                            # show the Qt Window


    def onSlider(self, value):
        self.imgActor.window(value*10) # change image contrast
        self.text2d.text(f"window level is now: {value*10}")
        self.plt.render()

    def onKeypress(self, evt):
        printc("You have pressed key:", evt.keyPressed, c='b')
        if evt.keyPressed=='q':
            self.plt.close()
            self.vtkWidget.close()
            exit()

    def onClose(self):
        self.vtkWidget.close()

if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.onClose)
    app.exec_()
