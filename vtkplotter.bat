@ECHO OFF
REM -------------------------------------------------
REM             Windows-10 users:
REM      Place this file on your desktop.
REM
REM  Double clicking it will open up a GUI,
REM  can drag&drop on icon to import files
REM
REM Set here the path to your Anaconda installation:
REM
SET anaconda_path=C:\ProgramData\anaconda3
REM
REM -------------------------------------------------

ECHO Activating Anaconda: %anaconda_path% ...
ECHO Starting vtkplotter...
CALL "%anaconda_path%\Scripts\activate"
python "%anaconda_path%\Scripts\vtkplotter" %*
ECHO Closing window...
REM PAUSE
