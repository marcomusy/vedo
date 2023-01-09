import sys
from PyQt5 import QtCore, QtWidgets
from vtk.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
from vedo import Mesh, dataurl, Plotter
from vedo.pyplot import np, plot

try:
    _encoding = QtWidgets.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtWidgets.QApplication.translate(context, text, disambig)


class Ui_MainWindow(object):

    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)

        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        self.tab1 = QtWidgets.QWidget()
        self.tab1.setObjectName("tab1")
        self.gridLayout1 = QtWidgets.QGridLayout(self.tab1)

        self.vtkLayout1 = QtWidgets.QVBoxLayout()
        self.vtkLayout1.setObjectName("vtkLayout1")
        self.gridLayout1.addLayout(self.vtkLayout1, 0, 0, 1, 1)

        self.tab2 = QtWidgets.QWidget()

        self.tabWidget = QtWidgets.QTabWidget(self.centralwidget)
        self.tabWidget.addTab(self.tab1, "tab1")
        self.tabWidget.addTab(self.tab2, "tab2")

        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.addWidget(self.tabWidget)

        self.gridLayout2 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout2.addLayout(self.verticalLayout, 0, 0, 1, 1)

        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 31))

        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        MainWindow.setStatusBar(self.statusbar)

        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow", None))
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab1), _translate("MainWindow", "Tab 1", None)
        )
        self.tabWidget.setTabText(
            self.tabWidget.indexOf(self.tab2), _translate("MainWindow", "Tab 2", None)
        )

        self.tabWidget.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


class MainWindow(QtWidgets.QMainWindow, Ui_MainWindow):

    def __init__(self, parent=None):
        QtWidgets.QMainWindow.__init__(self, parent)
        self.setupUi(self)

        self.vtkWidget1 = QVTKRenderWindowInteractor(self)
        self.vtkLayout1.addWidget(self.vtkWidget1)

        self.plt1 = Plotter(qt_widget=self.vtkWidget1, axes=1)
        self.id2 = self.plt1.add_callback("key press", self.onKeypress)
        self.plt1 += Mesh(dataurl+'shark.ply').c('cyan')
        self.plt1.show()

        self.vtkWidget2 = QVTKRenderWindowInteractor(self)
        self.verticalLayout.addWidget(self.vtkWidget2)
        self.plt2 = Plotter(qt_widget=self.vtkWidget2)

        #####################################################
        # add a plot using a formatted Figure
        x = np.random.randn(100) + 10
        y = np.random.randn(100) * 20 + 20
        fig = plot(
            x, y,
            lw=0,         # do not join points with lines
            xtitle="variable x",
            ytitle="variable y",
            marker="*",   # marker style
            mc="dr",      # marker color
            aspect=16/9,  # aspect ratio
        )
        self.plt2 += fig

        ##################################################### show
        self.plt2.show(zoom=1.8, mode='image')

    def onClose(self):
        self.vtkWidget1.close()

    def onKeypress(self, evt):
        print("You have pressed key:", evt.keypress)
        if evt.keypress=='q':
            sys.exit(0)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    app.aboutToQuit.connect(window.onClose)
    window.show()
    sys.exit(app.exec_())
