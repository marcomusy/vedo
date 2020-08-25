@ECHO OFF
REM ---------------------------------------------------------------
REM             Windows-10 users:
REM      Place this file on your desktop.
REM
REM  Double clicking it will open up a GUI,
REM  can drag&drop on icon to import files
REM
REM Set here the path to your Python/Anaconda installation, e.g.:
REM
SET anaconda_path=C:\ProgramData\anaconda3
REM
REM ---------------------------------------------------------------


REM ------------------------------------------
ECHO Activating Anaconda: %anaconda_path% ...
ECHO Starting vedo...
CALL "%anaconda_path%\Scripts\activate"
python "%anaconda_path%\Scripts\vedo" %*
ECHO Closing window...
REM PAUSE
