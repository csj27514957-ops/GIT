@echo off
setlocal
cd /d "%~dp0"
if errorlevel 1 goto failed
where powershell.exe >nul 2>&1
if errorlevel 1 (
    echo Windows PowerShell is required.
    goto failed
)
powershell.exe -NoLogo -NoProfile -File "%~dp0build_exe.ps1"
if errorlevel 1 goto failed
echo Build complete. Open the dist folder.
pause
exit /b 0

:failed
echo Build failed. Read the message above.
pause
exit /b 1
