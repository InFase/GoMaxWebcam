# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for GoMaxWebcam v2.

Cross-platform spec that works on Windows, macOS, and Linux.
Bundles all dashboard static assets, pyvirtualcam backends, and BLE support.

Usage:
    pyinstaller gomaxwebcam.spec

Platform-specific notes:
    Windows: Bundles UnityCaptureFilter DLLs if present
    macOS:   Creates .app bundle
    Linux:   Creates single-directory distribution
"""

import os
import sys
import platform
from pathlib import Path
from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

block_cipher = None

# Platform detection
IS_WINDOWS = sys.platform == 'win32'
IS_MACOS = sys.platform == 'darwin'
IS_LINUX = sys.platform.startswith('linux')

# Collect pyvirtualcam native backends
pyvirtualcam_binaries = collect_dynamic_libs('pyvirtualcam')

# Collect bleak data files (BLE UUID definitions etc.)
bleak_datas = collect_data_files('bleak')

# Dashboard static files (HTML, CSS, JS — all bundled locally)
dashboard_static = Path('src/gomaxwebcam/dashboard/static')
datas = [
    (str(dashboard_static), 'gomaxwebcam/dashboard/static'),
]
datas.extend(bleak_datas)

# Platform-specific data files
binaries = list(pyvirtualcam_binaries)
if IS_WINDOWS:
    unity_dir = Path('UnityCapture')
    if unity_dir.exists():
        for dll in unity_dir.glob('*.dll'):
            binaries.append((str(dll), 'UnityCapture'))

# Hidden imports for all platforms
hiddenimports = [
    # Core
    'gomaxwebcam',
    'gomaxwebcam.__main__',
    'gomaxwebcam.orchestrator',
    'gomaxwebcam.config',
    'gomaxwebcam.events',
    'gomaxwebcam.camera_manager',
    'gomaxwebcam.transport_manager',
    'gomaxwebcam.discovery',
    'gomaxwebcam.tray',
    'gomaxwebcam.logging_config',
    'gomaxwebcam.provisioning',
    'gomaxwebcam.failover',
    'gomaxwebcam.update_checker',
    # Dashboard
    'gomaxwebcam.dashboard',
    'gomaxwebcam.dashboard.app',
    'gomaxwebcam.dashboard.status_tracker',
    'gomaxwebcam.dashboard.wizard',
    'gomaxwebcam.dashboard.wizard_models',
    # Transports
    'gomaxwebcam.transport.base',
    'gomaxwebcam.transport.usb',
    'gomaxwebcam.transport.cohn',
    'gomaxwebcam.transport.wifi_ap',
    'gomaxwebcam.transport.wifi_manager',
    'gomaxwebcam.transport.cohn_orchestrator',
    'gomaxwebcam.transport.cohn_persistence',
    # BLE
    'gomaxwebcam.ble',
    'gomaxwebcam.ble.scanner',
    'gomaxwebcam.ble.gatt_client',
    'gomaxwebcam.ble.cohn',
    'gomaxwebcam.ble.uuids',
    # Pipeline
    'gomaxwebcam.pipeline',
    'gomaxwebcam.pipeline.frame_pipeline',
    'gomaxwebcam.pipeline.decode',
    'gomaxwebcam.pipeline.virtual_camera_sink',
    # Third-party
    'pyvirtualcam',
    'numpy',
    'psutil',
    'fastapi',
    'uvicorn',
    'uvicorn.logging',
    'uvicorn.protocols.http.auto',
    'uvicorn.protocols.http.h11_impl',
    'uvicorn.protocols.websockets.auto',
    'uvicorn.lifespan.on',
    'starlette',
    'httpx',
    'pystray',
    'PIL',
    'keyring',
    'keyring.backends',
    'platformdirs',
    'tomli_w',
    'bleak',
    'open_gopro',
]

# Platform-specific hidden imports
if IS_WINDOWS:
    hiddenimports.extend([
        'pyvirtualcam._native_windows_obs',
        'pyvirtualcam._native_windows_unity_capture',
        'keyring.backends.Windows',
    ])
elif IS_MACOS:
    hiddenimports.extend([
        'pyvirtualcam._native_macos_obs',
        'keyring.backends.macOS',
    ])
elif IS_LINUX:
    hiddenimports.extend([
        'pyvirtualcam._native_linux_v4l2',
        'keyring.backends.SecretService',
    ])

a = Analysis(
    ['src/gomaxwebcam/__main__.py'],
    pathex=['src'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'scipy',
        'pandas',
        'IPython',
        'jupyter',
        'notebook',
        'test',
        'unittest',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# Application name
APP_NAME = 'GoMaxWebcam'

if IS_MACOS:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        icon='assets/icon.icns' if Path('assets/icon.icns').exists() else None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=APP_NAME,
    )
    app = BUNDLE(
        coll,
        name=APP_NAME + '.app',
        icon='assets/icon.icns' if Path('assets/icon.icns').exists() else None,
        bundle_identifier='com.gomaxwebcam.app',
        info_plist={
            'CFBundleShortVersionString': '2.0.0',
            'NSBluetoothAlwaysUsageDescription': 'GoMaxWebcam uses Bluetooth to communicate with your GoPro camera.',
            'NSLocalNetworkUsageDescription': 'GoMaxWebcam needs local network access for camera streaming.',
        },
    )
else:
    exe = EXE(
        pyz,
        a.scripts,
        [],
        exclude_binaries=True,
        name=APP_NAME,
        debug=False,
        bootloader_ignore_signals=False,
        strip=False,
        upx=True,
        console=False,
        icon='assets/icon.ico' if Path('assets/icon.ico').exists() else None,
    )
    coll = COLLECT(
        exe,
        a.binaries,
        a.zipfiles,
        a.datas,
        strip=False,
        upx=True,
        upx_exclude=[],
        name=APP_NAME,
    )
