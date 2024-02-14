import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFrame, QVBoxLayout
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import dataurl, Volume, settings
from vedo.applications import Slicer3DPlotter


class MainWindow(QMainWindow):
    def __init__(self, parent=None):

        QMainWindow.__init__(self, parent)
        self.frame = QFrame()
        self.layout = QVBoxLayout()
        self.vtkWidget = QVTKRenderWindowInteractor(self.frame)

        # Create renderer and add the vedo objects and callbacks
        self.vol = Volume(dataurl + "embryo.slc")

        self.plt = Slicer3DPlotter(
            volume=self.vol,
            cmaps=("gist_ncar_r", "jet", "Spectral_r", "hot_r", "bone_r"),
            use_slider3d=True,
            bg="blue1",
            bg2="blue9",
            qt_widget=self.vtkWidget,
        )
        self.cid1 = self.plt.add_callback("mouse click", self._trigger)

        self.plt.show()

        # Set-up the rest of the Qt window
        self.layout.addWidget(self.vtkWidget)
        self.frame.setLayout(self.layout)
        self.setCentralWidget(self.frame)
        self.show()

    def _trigger(self, evt):
        # print("You have clicked your mouse button. Event info:\n", evt)
        i = int(self.plt.xslider.value)
        j = int(self.plt.yslider.value)
        k = int(self.plt.zslider.value)
        print(i,j,k, type(self.vol.xslice(i)))

    def onClose(self):
        # Disable the interactor before closing to prevent it
        # from trying to act on already deleted items
        print("CLOSING")
        self.vtkWidget.close()


if __name__ == "__main__":
    if settings.dry_run_mode:
        exit()
    app = QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.onClose)
    window.show()
    sys.exit(app.exec())