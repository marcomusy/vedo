import sys
from PyQt5 import Qt
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Plotter, Cone, printc

class MainWindow(Qt.QMainWindow):

    def __init__(self, parent=None):

        Qt.QMainWindow.__init__(self, parent)
        self.frame = Qt.QFrame()
        self.layout = Qt.QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        # Create renderer and add the vedo objects and callbacks
        self.plt = Plotter(qtWidget=self.vtkWidget)
        self.id1 = self.plt.addCallback("mouse click", self.onMouseClick)
        self.id2 = self.plt.addCallback("key press",   self.onKeypress)
        self.plt += Cone().rotateX(20)
        self.plt.show()                  # <--- show the vedo rendering

        # Set-up the rest of the Qt window
        button = Qt.QPushButton("My Button makes the cone red")
        button.setToolTip('This is an example button')
        button.clicked.connect(self.onClick)
        self.layout.addWidget(self.vtkWidget)
        self.layout.addWidget(button)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
        self.show()                     # <--- show the Qt Window

    def onMouseClick(self, evt):
        printc("You have clicked your mouse button. Event info:\n", evt, c='y')

    def onKeypress(self, evt):
        printc("You have pressed key:", evt.keyPressed, c='b')

    @Qt.pyqtSlot()
    def onClick(self):
        printc("..calling onClick")
        self.plt.actors[0].color('red').rotateZ(40)
        self.plt.interactor.Render()

    def onClose(self):
        #Disable the interactor before closing to prevent it
        #from trying to act on already deleted items
        printc("..calling onClose")
        self.vtkWidget.close()

if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.onClose) # <-- connect the onClose event
    app.exec_()
