@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto failed

if not exist "app.py" (
    echo app.py is missing. Extract the complete source ZIP first.
    goto failed
)

if exist ".venv\Scripts\python.exe" goto install
py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if errorlevel 1 goto try_python
py -3 -m venv ".venv"
if errorlevel 1 goto failed
goto install

:try_python
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 12) else 1)" >nul 2>&1
if errorlevel 1 (
    echo Install Python 3.12 or newer for Windows, including Tcl/Tk.
    echo https://www.python.org/downloads/windows/
    goto failed
)
python -m venv ".venv"
if errorlevel 1 goto failed

:install
".venv\Scripts\python.exe" -c "import sys, tkinter; sys.exit(0 if sys.version_info >= (3, 12) else 1)"
if errorlevel 1 (
    echo The local virtual environment needs Python 3.12 or newer with Tcl/Tk.
    echo Move the .venv folder aside, then run this file again.
    goto failed
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r "requirements.txt"
if errorlevel 1 goto failed
".venv\Scripts\python.exe" "app.py" %*
if errorlevel 1 goto failed
exit /b 0

:failed
echo FrameTag could not start. Read the message above.
pause
exit /b 1
