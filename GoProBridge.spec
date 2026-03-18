# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for GoPro Bridge."""

import os
import sys
from PyInstaller.utils.hooks import collect_dynamic_libs

block_cipher = None

# pyvirtualcam native backends
pyvirtualcam_binaries = collect_dynamic_libs('pyvirtualcam')

a = Analysis(
    ['src/main.py'],
    pathex=['src'],
    binaries=pyvirtualcam_binaries,
    datas=[
        ('UnityCapture/UnityCaptureFilter64.dll', 'UnityCapture'),
        ('UnityCapture/UnityCaptureFilter32.dll', 'UnityCapture'),
    ],
    hiddenimports=[
        'pyvirtualcam',
        'pyvirtualcam._native_windows_obs',
        'pyvirtualcam._native_windows_unity_capture',
        'psutil',
        'numpy',
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.sip',
        # app modules (loaded via internal imports)
        'config',
        'logger',
        'app_controller',
        'gopro_connection',
        'discovery',
        'stream_reader',
        'frame_pipeline',
        'frame_buffer',
        'virtual_camera',
        'firewall',
        'port_checker',
        'disconnect_detector',
        'usb_device_poller',
        'usb_event_listener',
        'gui',
        'utils',
        'stderr_ring_buffer',
        'dependency_checker',
        'setup_wizard',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'pandas',
        'PIL',
        'tkinter',
        'unittest',
        'pytest',
        'win32job',
        'win32api',
        'win32event',
        'win32process',
        'win32con',
    ],
    noarchive=False,
    optimize=0,
    cipher=block_cipher,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='GoProBridge',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='GoProBridge',
)
