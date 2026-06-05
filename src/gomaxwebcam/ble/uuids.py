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

# Network Management service UUID (b5f90090) contains two characteristics:
#   b5f90091 [write]  — request characteristic
#   b5f90092 [notify] — response characteristic

# Write network management requests (WiFi SSID, password, COHN config)
NETWORK_MGMT_REQUEST_UUID = "b5f90091-aa8d-11e3-9046-0002a5d5c51b"

# Read network management responses (Notify)
NETWORK_MGMT_RESPONSE_UUID = "b5f90092-aa8d-11e3-9046-0002a5d5c51b"

# ---------------------------------------------------------------------------
# Camera info (read-only)
# ---------------------------------------------------------------------------

# WiFi AP SSID (read)
WIFI_AP_SSID_UUID = "b5f90002-aa8d-11e3-9046-0002a5d5c51b"

# WiFi AP Password (read)
WIFI_AP_PASSWORD_UUID = "b5f90003-aa8d-11e3-9046-0002a5d5c51b"

# ---------------------------------------------------------------------------
# BLE Feature IDs (first byte of protobuf-wrapped BLE commands)
# ---------------------------------------------------------------------------

FEATURE_ID_COMMAND = 0xF1  # Write to Command characteristic (b5f90072)
FEATURE_ID_QUERY = 0xF5   # Write to Query characteristic (b5f90076)

# ---------------------------------------------------------------------------
# COHN Action IDs (from open_gopro.models.constants.ActionId)
# ---------------------------------------------------------------------------

COHN_CREATE_CERT = 0x67       # RequestCreateCOHNCert → Command char
COHN_CLEAR_CERT = 0x68        # RequestClearCOHNCert → Command char
COHN_GET_CERT = 0x69          # RequestCOHNCert → Command char
COHN_GET_STATUS = 0x6F        # RequestGetCOHNStatus → Query char
COHN_SET_SETTING = 0x70       # RequestCOHNSetting → Command char

# Response action IDs (in notification responses)
COHN_GET_STATUS_RESPONSE = 0xEF

# ---------------------------------------------------------------------------
# Network Management Feature + Action IDs (Network Mgmt characteristic)
# ---------------------------------------------------------------------------

FEATURE_ID_NETWORK_MGMT = 0x02

WIFI_SCAN_START = 0x02         # RequestStartScan → scan response + notification
WIFI_SCAN_RESULTS = 0x03       # RequestGetApEntries → list of WiFi networks
WIFI_CONNECT = 0x04            # RequestConnect → connect to provisioned AP
WIFI_CONNECT_NEW = 0x05        # RequestConnectNew → provision + connect new AP
WIFI_NOTIF_SCAN = 0x0B         # NotifStartScanning (notification)
WIFI_NOTIF_PROVIS = 0x0C       # NotifProvisioningState (notification)

# ---------------------------------------------------------------------------
# BLE advertisement filter constants
# ---------------------------------------------------------------------------

# GoPro manufacturer ID in BLE advertisements
GOPRO_MANUFACTURER_ID = 0x00D4  # 212 decimal — GoPro Inc.

# GoPro camera name prefixes in BLE advertisements
GOPRO_NAME_PREFIXES = ("GoPro ", "GP-")
