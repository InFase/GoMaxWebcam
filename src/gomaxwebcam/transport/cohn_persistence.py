"""
transport/cohn_persistence.py — COHN credential persistence via open-gopro's cohn_db.

Thin wrapper around open-gopro's TinyDB-backed credential store.
open-gopro owns COHN credential persistence (no keyring duplication).

This module provides:
  - store_credentials()     — Save credentials after BLE provisioning
  - load_credentials()      — Retrieve stored credentials for a camera
  - has_valid_credentials()  — Quick check if usable credentials exist
  - list_cameras()          — List all cameras with stored credentials
  - remove_credentials()    — Remove stored credentials for a camera
  - get_default_db_path()   — Standard path for the cohn_db.json file

Credential lifecycle:
  1. BLE provisioning (via BLEProvisioningService) provisions COHN
  2. open-gopro writes CohnInfo to cohn_db.json automatically
  3. This module reads from the same cohn_db.json for fast reconnect
  4. store_credentials() is used only when open-gopro doesn't auto-persist
     (e.g., manual credential entry from dashboard wizard)

Thread model:
    All methods are synchronous — they read/write a local JSON file.
    Safe to call from any thread or the asyncio event loop (via run_in_executor
    if blocking is a concern, though TinyDB is fast for small DBs).

Architecture:
    CohnCredentialStore
      └── TinyDB (cohn_db.json)
            └── open-gopro CohnDb adapter (search/insert operations)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger("gomaxwebcam.transport.cohn_persistence")


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------


def get_default_db_path() -> Path:
    """Return the standard path for the COHN credential database.

    Uses platformdirs for platform-appropriate config location:
      - Windows: %LOCALAPPDATA%/GoMaxWebcam-v2/cohn_db.json
      - macOS:   ~/Library/Application Support/GoMaxWebcam-v2/cohn_db.json
      - Linux:   ~/.config/GoMaxWebcam-v2/cohn_db.json
    """
    try:
        import platformdirs
        config_dir = Path(platformdirs.user_config_dir("GoMaxWebcam-v2"))
        config_dir.mkdir(parents=True, exist_ok=True)
        return config_dir / "cohn_db.json"
    except ImportError:
        log.warning("platformdirs not installed, using local cohn_db.json")
        return Path("cohn_db.json")


# ---------------------------------------------------------------------------
# Credential data class
# ---------------------------------------------------------------------------


@dataclass
class StoredCOHNCredentials:
    """COHN credentials retrieved from persistence.

    Mirrors the fields in open-gopro's CohnInfo but decoupled
    so callers don't need to import open-gopro types directly.

    Attributes:
        ip_address: Camera IP on the home network.
        username: HTTP Basic Auth username (typically 'gopro').
        password: HTTP Basic Auth password (device-generated).
        certificate: PEM-encoded TLS certificate for the camera.
        camera_serial: Camera serial suffix (last 4 digits, used as DB key).
    """
    ip_address: str
    username: str
    password: str
    certificate: str
    camera_serial: str

    @property
    def is_valid(self) -> bool:
        """True if all required credential fields are populated.

        A credential set is valid when it has enough information to
        construct a CohnInfo and connect via COHN HTTPS without BLE.
        """
        return bool(
            self.ip_address
            and self.username
            and self.password
            and self.certificate
        )

    def to_dict(self) -> dict:
        """Serialize to dict for JSON/API responses."""
        return {
            "ip_address": self.ip_address,
            "username": self.username,
            "password": self.password,
            "certificate": self.certificate,
            "camera_serial": self.camera_serial,
            "is_valid": self.is_valid,
        }


# ---------------------------------------------------------------------------
# Credential store
# ---------------------------------------------------------------------------


class CohnCredentialStore:
    """Persistence layer for COHN credentials using open-gopro's cohn_db.

    Wraps TinyDB operations to store, retrieve, and validate COHN
    credentials. Uses the same database file that open-gopro writes to
    during BLE provisioning, so credentials from either source are
    seamlessly available.

    Args:
        db_path: Path to the TinyDB JSON file. Defaults to the
            platform-standard config location.

    Usage:
        store = CohnCredentialStore()

        # Check for existing credentials
        if store.has_valid_credentials("1234"):
            creds = store.load_credentials("1234")
            print(f"Camera at {creds.ip_address}")

        # Store new credentials (manual entry or post-provision)
        store.store_credentials(StoredCOHNCredentials(
            ip_address="192.168.1.100",
            username="gopro",
            password="s3cret",
            certificate="-----BEGIN CERTIFICATE-----\\n...",
            camera_serial="1234",
        ))

        # List all cameras
        for creds in store.list_cameras():
            print(f"{creds.camera_serial}: {creds.ip_address}")
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = db_path or get_default_db_path()
        log.debug("CohnCredentialStore using db: %s", self._db_path)

    @property
    def db_path(self) -> Path:
        """Path to the underlying TinyDB file."""
        return self._db_path

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def load_credentials(self, camera_serial: str) -> Optional[StoredCOHNCredentials]:
        """Load stored COHN credentials for a camera.

        Reads from open-gopro's cohn_db.json (TinyDB). Does NOT
        contact the camera — purely a local database lookup.

        Args:
            camera_serial: Camera serial suffix (last 4 digits).

        Returns:
            StoredCOHNCredentials if found and valid, None otherwise.
        """
        if not camera_serial:
            log.debug("load_credentials called with empty serial")
            return None

        try:
            db, cohn_db = self._open_db()
            try:
                info = cohn_db.search_credentials(camera_serial)
                if info is None:
                    log.debug("No credentials found for serial %s", camera_serial)
                    return None

                creds = StoredCOHNCredentials(
                    ip_address=getattr(info, "ip_address", "") or "",
                    username=getattr(info, "username", "") or "",
                    password=getattr(info, "password", "") or "",
                    certificate=getattr(info, "certificate", "") or "",
                    camera_serial=camera_serial,
                )

                if creds.is_valid:
                    log.info(
                        "Loaded valid credentials for serial %s (ip=%s)",
                        camera_serial, creds.ip_address,
                    )
                    return creds
                else:
                    log.warning(
                        "Credentials for serial %s are incomplete", camera_serial,
                    )
                    return None
            finally:
                db.close()

        except ImportError:
            log.warning(
                "Cannot load credentials: tinydb or open-gopro not installed"
            )
            return None
        except Exception as e:
            log.warning("Failed to load credentials for %s: %s", camera_serial, e)
            return None

    def store_credentials(self, credentials: StoredCOHNCredentials) -> bool:
        """Store COHN credentials in the cohn_db.

        Typically open-gopro persists credentials automatically during
        BLE provisioning. This method is for cases where credentials
        need to be stored manually (e.g., from dashboard wizard input
        or imported from another source).

        Args:
            credentials: The credentials to store.

        Returns:
            True if stored successfully, False on error.
        """
        if not credentials.camera_serial:
            log.error("Cannot store credentials without camera_serial")
            return False

        try:
            db, cohn_db = self._open_db()
            try:
                # Build a CohnInfo-compatible object for the DB
                from open_gopro.models.general import CohnInfo

                cohn_info = CohnInfo(
                    ip_address=credentials.ip_address,
                    username=credentials.username,
                    password=credentials.password,
                    certificate=credentials.certificate,
                )

                # Use the cohn_db's insert method
                # open-gopro's CohnDb stores by camera serial
                cohn_db.insert_or_update_credentials(
                    credentials.camera_serial, cohn_info,
                )

                log.info(
                    "Stored credentials for serial %s (ip=%s)",
                    credentials.camera_serial, credentials.ip_address,
                )
                return True
            finally:
                db.close()

        except ImportError:
            log.warning(
                "Cannot store credentials: tinydb or open-gopro not installed"
            )
            return False
        except Exception as e:
            log.error(
                "Failed to store credentials for %s: %s",
                credentials.camera_serial, e,
            )
            return False

    def has_valid_credentials(self, camera_serial: str) -> bool:
        """Check if valid COHN credentials exist for a camera.

        Quick boolean check — loads credentials and validates them
        without returning the full credential object.

        Args:
            camera_serial: Camera serial suffix (last 4 digits).

        Returns:
            True if valid, complete credentials are stored.
        """
        creds = self.load_credentials(camera_serial)
        return creds is not None and creds.is_valid

    def list_cameras(self) -> list[StoredCOHNCredentials]:
        """List all cameras with stored COHN credentials.

        Returns all credential entries from the database, including
        incomplete ones (check .is_valid on each).

        Returns:
            List of StoredCOHNCredentials for all known cameras.
        """
        try:
            db, cohn_db = self._open_db()
            try:
                results: list[StoredCOHNCredentials] = []
                # TinyDB: iterate all documents in the default table
                for doc in db.all():
                    serial = doc.get("serial", "") or doc.get("camera_serial", "")
                    ip = doc.get("ip_address", "")
                    username = doc.get("username", "")
                    password = doc.get("password", "")
                    certificate = doc.get("certificate", "")

                    if serial:
                        results.append(StoredCOHNCredentials(
                            ip_address=ip,
                            username=username,
                            password=password,
                            certificate=certificate,
                            camera_serial=serial,
                        ))

                log.debug("Listed %d cameras from cohn_db", len(results))
                return results
            finally:
                db.close()

        except ImportError:
            log.warning(
                "Cannot list cameras: tinydb or open-gopro not installed"
            )
            return []
        except Exception as e:
            log.warning("Failed to list cameras: %s", e)
            return []

    def invalidate_credentials(self, camera_serial: str) -> bool:
        """Mark stored COHN credentials as invalid (stale/expired).

        Removes the credentials from the store so they are not reused on the
        next eager reconnect attempt.  This is called when a COHN connection
        attempt fails with an authentication or TLS error, indicating that
        the cached credentials are stale (e.g., camera was factory-reset,
        COHN was re-provisioned from a different client, or the camera's
        certificate was rotated).

        Equivalent to ``remove_credentials`` but semantically distinct:
        invalidate signals that re-provisioning via BLE is needed, while
        remove is a general-purpose deletion.

        Args:
            camera_serial: Camera serial suffix (last 4 digits).

        Returns:
            True if credentials were found and removed, False otherwise.
        """
        if not camera_serial:
            log.debug("invalidate_credentials called with empty serial")
            return False

        log.info(
            "Invalidating cached COHN credentials for serial %s "
            "(will require BLE re-provisioning)",
            camera_serial,
        )
        return self.remove_credentials(camera_serial)

    def remove_credentials(self, camera_serial: str) -> bool:
        """Remove stored COHN credentials for a camera.

        Args:
            camera_serial: Camera serial suffix (last 4 digits).

        Returns:
            True if credentials were found and removed, False otherwise.
        """
        if not camera_serial:
            return False

        try:
            from tinydb import TinyDB, where

            db = TinyDB(str(self._db_path), indent=4)
            try:
                removed = db.remove(where("serial") == camera_serial)
                if not removed:
                    # Try alternate key name
                    removed = db.remove(
                        where("camera_serial") == camera_serial
                    )

                if removed:
                    log.info("Removed credentials for serial %s", camera_serial)
                    return True
                else:
                    log.debug(
                        "No credentials found to remove for serial %s",
                        camera_serial,
                    )
                    return False
            finally:
                db.close()

        except ImportError:
            log.warning("Cannot remove credentials: tinydb not installed")
            return False
        except Exception as e:
            log.warning(
                "Failed to remove credentials for %s: %s", camera_serial, e,
            )
            return False

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    def to_cohn_info(self, credentials: StoredCOHNCredentials) -> Any:
        """Convert StoredCOHNCredentials to open-gopro CohnInfo.

        Used when constructing a WirelessGoPro with COHN interface.

        Args:
            credentials: Stored credentials to convert.

        Returns:
            CohnInfo instance, or None if open-gopro is not installed.
        """
        try:
            from open_gopro.models.general import CohnInfo

            return CohnInfo(
                ip_address=credentials.ip_address,
                username=credentials.username,
                password=credentials.password,
                certificate=credentials.certificate,
            )
        except ImportError:
            log.warning("Cannot create CohnInfo: open-gopro not installed")
            return None

    @staticmethod
    def from_provision_result(
        ip_address: str,
        username: str,
        password: str,
        certificate: str,
        camera_serial: str,
    ) -> StoredCOHNCredentials:
        """Create StoredCOHNCredentials from raw provisioning results.

        Convenience factory for converting provisioning output
        (from COHNOrchestrator or BLEProvisioningService) into
        the persistence-layer credential type.
        """
        return StoredCOHNCredentials(
            ip_address=ip_address,
            username=username,
            password=password,
            certificate=certificate,
            camera_serial=camera_serial,
        )

    # ------------------------------------------------------------------
    # Internal: DB access
    # ------------------------------------------------------------------

    def _open_db(self) -> tuple:
        """Open TinyDB and wrap with open-gopro's CohnDb adapter.

        Returns:
            Tuple of (TinyDB instance, CohnDb adapter).
            Caller must close the TinyDB instance.

        Raises:
            ImportError: If tinydb or open-gopro is not installed.
        """
        from tinydb import TinyDB
        from open_gopro.database.cohn_db import CohnDb

        # Ensure parent directory exists
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        db = TinyDB(str(self._db_path), indent=4)
        cohn_db = CohnDb(db)
        return db, cohn_db
