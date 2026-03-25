"""
ble/uuids.py — GoPro BLE GATT service and characteristic UUIDs.

GoPro cameras advertise a custom BLE service and expose GATT characteristics
for camera control and COHN (Camera On Home Network) provisioning.

UUIDs sourced from:
  - open-gopro BLE specification
  - GoPro Open API documentation

Hero 11/12/13 and MAX 2 share the same BLE service UUID and COHN
provisioning characteristics.
"""

# ---------------------------------------------------------------------------
# GoPro BLE service UUID — advertised by all supported models
# ---------------------------------------------------------------------------

GOPRO_SERVICE_UUID = "0000fea6-0000-1000-8000-00805f9b34fb"

# ---------------------------------------------------------------------------
# Command / Response characteristics (GP-xxxx)
# ---------------------------------------------------------------------------

# Write commands to the camera (Request)
COMMAND_REQUEST_UUID = "b5f90072-aa8d-11e3-9046-0002a5d5c51b"

# Read command responses from the camera (Notify)
COMMAND_RESPONSE_UUID = "b5f90073-aa8d-11e3-9046-0002a5d5c51b"

# Write setting changes to the camera (Request)
SETTINGS_REQUEST_UUID = "b5f90074-aa8d-11e3-9046-0002a5d5c51b"

# Read setting responses from the camera (Notify)
SETTINGS_RESPONSE_UUID = "b5f90075-aa8d-11e3-9046-0002a5d5c51b"

# Query status/settings from the camera (Request)
QUERY_REQUEST_UUID = "b5f90076-aa8d-11e3-9046-0002a5d5c51b"

# Read query responses from the camera (Notify)
QUERY_RESPONSE_UUID = "b5f90077-aa8d-11e3-9046-0002a5d5c51b"

# ---------------------------------------------------------------------------
# Network Management characteristics (for WiFi + COHN provisioning)
# ---------------------------------------------------------------------------

# Write network management requests (WiFi SSID, password, COHN config)
NETWORK_MGMT_REQUEST_UUID = "b5f90090-aa8d-11e3-9046-0002a5d5c51b"

# Read network management responses (Notify)
NETWORK_MGMT_RESPONSE_UUID = "b5f90091-aa8d-11e3-9046-0002a5d5c51b"

# ---------------------------------------------------------------------------
# Camera info (read-only)
# ---------------------------------------------------------------------------

# WiFi AP SSID (read)
WIFI_AP_SSID_UUID = "b5f90002-aa8d-11e3-9046-0002a5d5c51b"

# WiFi AP Password (read)
WIFI_AP_PASSWORD_UUID = "b5f90003-aa8d-11e3-9046-0002a5d5c51b"

# ---------------------------------------------------------------------------
# Protobuf command IDs for COHN provisioning (feature IDs in request TLVs)
# ---------------------------------------------------------------------------

# These are the protobuf feature IDs used in the network management
# request/response protocol for COHN provisioning.
COHN_FEATURE_ID = 0xF5  # COHN-related commands

# COHN request action IDs (sub-commands within the COHN feature)
COHN_GET_STATUS = 0x01
COHN_SET_SETTING = 0x02
COHN_CREATE_CERT = 0x03
COHN_CLEAR_CERT = 0x04
COHN_GET_CERT = 0x05

# WiFi scan/connect action IDs (used for station-mode WiFi provisioning)
WIFI_SCAN_START = 0x02
WIFI_SCAN_RESULTS = 0x03
WIFI_CONNECT = 0x04
WIFI_GET_STATUS = 0x05

# ---------------------------------------------------------------------------
# BLE advertisement filter constants
# ---------------------------------------------------------------------------

# GoPro manufacturer ID in BLE advertisements
GOPRO_MANUFACTURER_ID = 0x00D4  # 212 decimal — GoPro Inc.

# GoPro camera name prefixes in BLE advertisements
GOPRO_NAME_PREFIXES = ("GoPro ", "GP-")
