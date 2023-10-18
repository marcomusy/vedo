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
        self.plt = Plotter(qt_widget=self.vtkWidget)
        self.plt += Cone().rotate_x(20)

        self.button = self.plt.add_button(
            self.buttonfunc,
            pos=(0.7, 0.05),  # x,y fraction from bottom left corner
            states=["click to green"],  # text for each state
            c=["w"],  # font color for each state
            bc=["dg"],  # background color for each state
            font="courier",  # font type
            size=25,  # font size
            bold=True,  # bold font
            italic=False,  # non-italic font style
        )

        self.plt.show()  # <--- show the vedo rendering

        # Set-up the rest of the Qt window
        button = Qt.QPushButton("My Button makes the cone red")
        button.setToolTip("This is an example button")
        button.clicked.connect(self.onClick)
        self.layout.addWidget(self.vtkWidget)
        self.layout.addWidget(button)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
        self.show()  # <--- show the Qt Window

    def buttonfunc(self, obj, ename):
        print("btn is clicked...")
        self.plt.objects[0].color("green5").rotate_z(40)
        
    @Qt.pyqtSlot()
    def onClick(self):
        printc("..calling onClick")
        self.plt.objects[0].color("red5").rotate_z(40)
        self.plt.render()


if __name__ == "__main__":
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    app.exec_()