@echo off
title GoMaxWebcam Uninstaller
echo.
echo GoMaxWebcam Uninstaller
echo =======================
echo.

:: Find Python
where py >nul 2>&1 && (py -3 install.py --uninstall & goto :done)
where python3 >nul 2>&1 && (python3 install.py --uninstall & goto :done)
where python >nul 2>&1 && (python install.py --uninstall & goto :done)

echo ERROR: Python not found. Please uninstall manually:
echo   pip uninstall gomaxwebcam
echo   Delete desktop shortcuts manually.

:done
echo.
pause
