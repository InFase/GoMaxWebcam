"""
Tests for the rotating log handler with session-based cleanup policy.

Covers:
  - Session file creation per setup_logger call
  - Cleanup by session count (max 5 files)
  - Cleanup by total size cap (50 MB)
  - Whichever-comes-first behavior
  - Old file deletion order (oldest removed first)
  - Rotated chunk cleanup alongside session files
  - Edge cases: empty dir, single file, exact limits
"""

import logging
import os
import time
from pathlib import Path

import pytest

# Adjust import path for src/ layout
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from logger import (

    cleanup_logs,
    get_logger,
    reset_logger,
    setup_logger,
    _get_session_log_files,
    _get_all_log_files,
    _total_log_size,
    _cleanup_by_session_count,
    _cleanup_by_total_size,
)

pytestmark = pytest.mark.no_gopro_needed



@pytest.fixture
def log_dir(tmp_path):
    """Provide a clean temporary log directory."""
    d = tmp_path / "logs"
    d.mkdir()
    return d


@pytest.fixture(autouse=True)
def _reset_logger():
    """Reset logger state before and after each test."""
    reset_logger()
    yield
    reset_logger()


# ---------------------------------------------------------------------------
# Helper to create fake session log files with controlled size and mtime
# ---------------------------------------------------------------------------

def _create_fake_log(log_dir: Path, name: str, size_bytes: int = 100, age_offset: float = 0.0) -> Path:
    """Create a fake log file with given size. age_offset is subtracted from current time."""
    p = log_dir / name
    p.write_bytes(b"X" * size_bytes)
    if age_offset > 0:
        mtime = time.time() - age_offset
        os.utime(p, (mtime, mtime))
    return p


# ---------------------------------------------------------------------------
# Tests: _get_session_log_files
# ---------------------------------------------------------------------------

class TestGetSessionLogFiles:
    def test_returns_empty_for_empty_dir(self, log_dir):
        assert _get_session_log_files(log_dir) == []

    def test_returns_only_session_files(self, log_dir):
        _create_fake_log(log_dir, "gopro_bridge_20250101_120000.log")
        _create_fake_log(log_dir, "gopro_bridge_20250101_120000.log.1")
        _create_fake_log(log_dir, "other_file.log")

        result = _get_session_log_files(log_dir)
        names = [f.name for f in result]
        assert names == ["gopro_bridge_20250101_120000.log"]

    def test_sorted_oldest_first(self, log_dir):
        _create_fake_log(log_dir, "gopro_bridge_20250102.log", age_offset=0)
        _create_fake_log(log_dir, "gopro_bridge_20250101.log", age_offset=10)

        result = _get_session_log_files(log_dir)
        assert result[0].name == "gopro_bridge_20250101.log"
        assert result[1].name == "gopro_bridge_20250102.log"


# ---------------------------------------------------------------------------
# Tests: _get_all_log_files
# ---------------------------------------------------------------------------

class TestGetAllLogFiles:
    def test_includes_chunks(self, log_dir):
        _create_fake_log(log_dir, "gopro_bridge_20250101.log")
        _create_fake_log(log_dir, "gopro_bridge_20250101.log.1")
        _create_fake_log(log_dir, "gopro_bridge_20250101.log.2")
        _create_fake_log(log_dir, "unrelated.txt")

        result = _get_all_log_files(log_dir)
        assert len(result) == 3
        assert all("gopro_bridge" in f.name for f in result)


# ---------------------------------------------------------------------------
# Tests: _total_log_size
# ---------------------------------------------------------------------------

class TestTotalLogSize:
    def test_sums_all_log_files(self, log_dir):
        _create_fake_log(log_dir, "gopro_bridge_a.log", size_bytes=1000)
        _create_fake_log(log_dir, "gopro_bridge_b.log", size_bytes=2000)
        _create_fake_log(log_dir, "gopro_bridge_b.log.1", size_bytes=500)

        assert _total_log_size(log_dir) == 3500

    def test_ignores_non_log_files(self, log_dir):
        _create_fake_log(log_dir, "gopro_bridge_a.log", size_bytes=100)
        _create_fake_log(log_dir, "readme.txt", size_bytes=9999)

        assert _total_log_size(log_dir) == 100


# ---------------------------------------------------------------------------
# Tests: _cleanup_by_session_count
# ---------------------------------------------------------------------------

class TestCleanupBySessionCount:
    def test_no_removal_under_limit(self, log_dir):
        for i in range(3):
            _create_fake_log(log_dir, f"gopro_bridge_{i:04d}.log", age_offset=10 - i)

        removed = _cleanup_by_session_count(log_dir, max_sessions=5)
        assert removed == 0
        assert len(_get_session_log_files(log_dir)) == 3

    def test_removes_oldest_when_over_limit(self, log_dir):
        for i in range(7):
            _create_fake_log(log_dir, f"gopro_bridge_{i:04d}.log", age_offset=70 - i * 10)

        removed = _cleanup_by_session_count(log_dir, max_sessions=5)
        assert removed == 2
        remaining = _get_session_log_files(log_dir)
        assert len(remaining) == 5
        # The two oldest (0000, 0001) should be gone
        names = {f.name for f in remaining}
        assert "gopro_bridge_0000.log" not in names
        assert "gopro_bridge_0001.log" not in names

    def test_removes_rotated_chunks_with_session(self, log_dir):
        _create_fake_log(log_dir, "gopro_bridge_old.log", age_offset=100)
        _create_fake_log(log_dir, "gopro_bridge_old.log.1", age_offset=100)
        _create_fake_log(log_dir, "gopro_bridge_old.log.2", age_offset=100)
        _create_fake_log(log_dir, "gopro_bridge_new.log", age_offset=0)

        removed = _cleanup_by_session_count(log_dir, max_sessions=1)
        assert removed == 3  # old.log + old.log.1 + old.log.2
        assert len(list(log_dir.glob("gopro_bridge_old*"))) == 0
        assert (log_dir / "gopro_bridge_new.log").exists()

    def test_exact_limit_no_removal(self, log_dir):
        for i in range(5):
            _create_fake_log(log_dir, f"gopro_bridge_{i:04d}.log", age_offset=50 - i * 10)

        removed = _cleanup_by_session_count(log_dir, max_sessions=5)
        assert removed == 0


# ---------------------------------------------------------------------------
# Tests: _cleanup_by_total_size
# ---------------------------------------------------------------------------

class TestCleanupByTotalSize:
    def test_no_removal_under_cap(self, log_dir):
        _create_fake_log(log_dir, "gopro_bridge_a.log", size_bytes=1000)
        removed = _cleanup_by_total_size(log_dir, max_bytes=5000)
        assert removed == 0

    def test_removes_oldest_until_under_cap(self, log_dir):
        _create_fake_log(log_dir, "gopro_bridge_old.log", size_bytes=3000, age_offset=20)
        _create_fake_log(log_dir, "gopro_bridge_mid.log", size_bytes=3000, age_offset=10)
        _create_fake_log(log_dir, "gopro_bridge_new.log", size_bytes=3000, age_offset=0)

        removed = _cleanup_by_total_size(log_dir, max_bytes=7000)
        assert removed == 1  # removes oldest to get from 9000 to 6000
        assert not (log_dir / "gopro_bridge_old.log").exists()
        assert (log_dir / "gopro_bridge_new.log").exists()

    def test_removes_multiple_until_under(self, log_dir):
        for i in range(5):
            _create_fake_log(log_dir, f"gopro_bridge_{i:04d}.log", size_bytes=2000, age_offset=50 - i * 10)

        # Total = 10000, cap = 4500 → need to remove 3 files
        removed = _cleanup_by_total_size(log_dir, max_bytes=4500)
        assert removed == 3
        assert _total_log_size(log_dir) <= 4500


# ---------------------------------------------------------------------------
# Tests: cleanup_logs (combined policy)
# ---------------------------------------------------------------------------

class TestCleanupLogs:
    def test_session_count_triggers_first(self, log_dir):
        """Session count limit (3) kicks in before size limit."""
        for i in range(6):
            _create_fake_log(log_dir, f"gopro_bridge_{i:04d}.log", size_bytes=100, age_offset=60 - i * 10)

        removed = cleanup_logs(log_dir, max_sessions=3, max_total_bytes=999999)
        assert removed == 3
        assert len(_get_session_log_files(log_dir)) == 3

    def test_size_cap_triggers_after_session_cleanup(self, log_dir):
        """Even with 5 session files, if total > cap, more get removed."""
        for i in range(5):
            _create_fake_log(log_dir, f"gopro_bridge_{i:04d}.log", size_bytes=5000, age_offset=50 - i * 10)

        # Total = 25000, cap = 12000
        removed = cleanup_logs(log_dir, max_sessions=5, max_total_bytes=12000)
        assert removed >= 2
        assert _total_log_size(log_dir) <= 12000

    def test_nonexistent_dir_returns_zero(self, tmp_path):
        missing = tmp_path / "does_not_exist"
        assert cleanup_logs(missing) == 0

    def test_empty_dir_returns_zero(self, log_dir):
        assert cleanup_logs(log_dir) == 0


# ---------------------------------------------------------------------------
# Tests: setup_logger integration
# ---------------------------------------------------------------------------

class TestSetupLogger:
    def test_creates_session_log_file(self, log_dir):
        logger = setup_logger(log_dir, level="DEBUG")
        session_files = _get_session_log_files(log_dir)
        assert len(session_files) == 1
        assert session_files[0].name.startswith("gopro_bridge_")

    def test_second_call_returns_same_logger(self, log_dir):
        logger1 = setup_logger(log_dir)
        logger2 = setup_logger(log_dir)
        assert logger1 is logger2

    def test_cleans_up_old_files_on_init(self, log_dir):
        # Pre-create 6 old session files
        for i in range(6):
            _create_fake_log(log_dir, f"gopro_bridge_{i:04d}.log", size_bytes=100, age_offset=60 - i * 10)

        logger = setup_logger(log_dir, max_session_files=3)
        # 3 oldest removed by cleanup, then 1 new created = 4 total
        session_files = _get_session_log_files(log_dir)
        assert len(session_files) == 4  # 3 kept + 1 new

    def test_log_level_applied(self, log_dir):
        logger = setup_logger(log_dir, level="WARNING")
        assert logger.level == logging.WARNING

    def test_logger_writes_to_file(self, log_dir):
        logger = setup_logger(log_dir, level="DEBUG")
        logger.info("test message 12345")

        # Flush handlers
        for h in logger.handlers:
            h.flush()

        session_files = _get_session_log_files(log_dir)
        content = session_files[0].read_text(encoding="utf-8")
        assert "test message 12345" in content

    def test_cleanup_with_size_cap(self, log_dir):
        # Create files that total over 500 bytes
        for i in range(5):
            _create_fake_log(log_dir, f"gopro_bridge_{i:04d}.log", size_bytes=200, age_offset=50 - i * 10)

        logger = setup_logger(log_dir, max_session_files=10, max_total_bytes=500)
        # Should have cleaned up enough files to get under 500 bytes
        # (plus the new session file)
        assert _total_log_size(log_dir) <= 500 + 1024  # allow for new session header


# ---------------------------------------------------------------------------
# Tests: get_logger
# ---------------------------------------------------------------------------

class TestGetLogger:
    def test_returns_child_logger(self, log_dir):
        setup_logger(log_dir)
        child = get_logger("discovery")
        assert child.name == "gopro_bridge.discovery"

    def test_child_inherits_handlers(self, log_dir):
        parent = setup_logger(log_dir, level="DEBUG")
        child = get_logger("test_module")
        child.info("child message xyz")

        for h in parent.handlers:
            h.flush()

        session_files = _get_session_log_files(log_dir)
        content = session_files[0].read_text(encoding="utf-8")
        assert "child message xyz" in content


# ---------------------------------------------------------------------------
# Tests: reset_logger
# ---------------------------------------------------------------------------

class TestResetLogger:
    def test_allows_reinitalization(self, log_dir):
        logger1 = setup_logger(log_dir)
        handler_count1 = len(logger1.handlers)

        reset_logger()

        log_dir2 = log_dir.parent / "logs2"
        log_dir2.mkdir()
        logger2 = setup_logger(log_dir2)

        # Should have fresh handlers, not accumulated
        assert len(logger2.handlers) == handler_count1
