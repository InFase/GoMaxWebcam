@echo off
:: GoMaxWebcam Installer — double-click to run setup
:: Finds Python automatically and runs install.py

setlocal

:: Try py launcher first (official Python installer sets this up)
where py >nul 2>&1
if %errorlevel% equ 0 (
    py -3 "%~dp0install.py" %*
    goto :done
)

:: Try python3
where python3 >nul 2>&1
if %errorlevel% equ 0 (
    python3 "%~dp0install.py" %*
    goto :done
)

:: Try python
where python >nul 2>&1
if %errorlevel% equ 0 (
    python "%~dp0install.py" %*
    goto :done
)

echo.
echo  ERROR: Python not found.
echo  Download Python 3.12+ from https://www.python.org/downloads/
echo  Make sure to check "Add Python to PATH" during installation.
echo.

:done
pause
