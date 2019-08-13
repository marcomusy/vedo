@ECHO OFF
REM ---------------------------------------------
REM             Windows-10 users
REM Place this file on your desktop.
REM (double clicking it will open up a GUI)
REM
REM Set the path to where Anaconda3 is installed:
SET anaconda_path="C:\ProgramData\anaconda3"
REM ---------------------------------------------


ECHO Starting vtkplotter...
CALL "%anaconda_path%\Scripts\activate"
python "%anaconda_path%\Scripts\vtkplotter" %*

ECHO Closing window...
REM pause