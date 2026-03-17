"""
test_stderr_ring_buffer.py — Tests for the thread-safe ffmpeg stderr ring buffer

Covers:
  - Basic write/read operations
  - Line count limit enforcement
  - Byte size limit enforcement
  - Thread safety under concurrent writes
  - Chunk parsing (\\r and \\n normalization)
  - Error line filtering
  - Clear operation
  - Edge cases (empty buffer, oversized lines)
"""

import threading

import pytest
from stderr_ring_buffer import StderrRingBuffer

pytestmark = pytest.mark.no_gopro_needed


class TestStderrRingBufferBasic:
    """Basic write and read operations."""

    def test_write_and_read_single_line(self):
        buf = StderrRingBuffer()
        buf.write_line("frame=  100 fps=30")
        assert buf.get_lines() == ["frame=  100 fps=30"]

    def test_write_multiple_lines(self):
        buf = StderrRingBuffer()
        buf.write_line("line 1")
        buf.write_line("line 2")
        buf.write_line("line 3")
        assert buf.get_lines() == ["line 1", "line 2", "line 3"]

    def test_get_text_joins_with_newlines(self):
        buf = StderrRingBuffer()
        buf.write_line("alpha")
        buf.write_line("beta")
        assert buf.get_text() == "alpha\nbeta"

    def test_get_last_line(self):
        buf = StderrRingBuffer()
        buf.write_line("first")
        buf.write_line("second")
        assert buf.get_last_line() == "second"

    def test_get_last_line_empty_buffer(self):
        buf = StderrRingBuffer()
        assert buf.get_last_line() == ""

    def test_line_count_property(self):
        buf = StderrRingBuffer()
        assert buf.line_count == 0
        buf.write_line("a")
        buf.write_line("b")
        assert buf.line_count == 2

    def test_total_bytes_property(self):
        buf = StderrRingBuffer()
        buf.write_line("hello")  # 5 bytes
        assert buf.total_bytes == 5

    def test_clear(self):
        buf = StderrRingBuffer()
        buf.write_line("data")
        buf.clear()
        assert buf.line_count == 0
        assert buf.total_bytes == 0
        assert buf.get_lines() == []


class TestStderrRingBufferLimits:
    """Line count and byte size limit enforcement."""

    def test_line_count_eviction(self):
        buf = StderrRingBuffer(max_lines=3, max_bytes=1024 * 1024)
        for i in range(5):
            buf.write_line(f"line {i}")
        assert buf.line_count == 3
        # Should have the 3 newest lines
        assert buf.get_lines() == ["line 2", "line 3", "line 4"]

    def test_byte_size_eviction(self):
        # Each line is 10 bytes ("aaaaaaaaaa"), cap at 25 bytes => max 2 lines
        buf = StderrRingBuffer(max_lines=1000, max_bytes=25)
        buf.write_line("a" * 10)  # 10 bytes
        buf.write_line("b" * 10)  # 20 bytes total
        buf.write_line("c" * 10)  # 30 bytes -> evict oldest
        assert buf.line_count == 2
        lines = buf.get_lines()
        assert lines[0] == "b" * 10
        assert lines[1] == "c" * 10
        assert buf.total_bytes == 20

    def test_oversized_single_line_truncated(self):
        buf = StderrRingBuffer(max_lines=100, max_bytes=50)
        buf.write_line("x" * 200)
        # Line should be truncated to max_bytes
        assert buf.line_count == 1
        assert buf.total_bytes <= 50

    def test_default_limits(self):
        buf = StderrRingBuffer()
        assert buf.max_lines == 1000
        assert buf.max_bytes == 256 * 1024


class TestStderrRingBufferChunkParsing:
    """Chunk parsing with \\r and \\n normalization."""

    def test_write_chunk_newlines(self):
        buf = StderrRingBuffer()
        buf.write_chunk("line1\nline2\nline3\n")
        assert buf.get_lines() == ["line1", "line2", "line3"]

    def test_write_chunk_carriage_returns(self):
        buf = StderrRingBuffer()
        buf.write_chunk("frame=1\rframe=2\rframe=3")
        assert buf.get_lines() == ["frame=1", "frame=2", "frame=3"]

    def test_write_chunk_mixed_separators(self):
        buf = StderrRingBuffer()
        buf.write_chunk("progress\r\n[error] bad\nok")
        assert buf.get_lines() == ["progress", "[error] bad", "ok"]

    def test_write_chunk_skips_empty(self):
        buf = StderrRingBuffer()
        buf.write_chunk("\n\n\nhello\n\n")
        assert buf.get_lines() == ["hello"]


class TestStderrRingBufferErrorFiltering:
    """Error keyword filtering."""

    def test_get_error_lines(self):
        buf = StderrRingBuffer()
        buf.write_line("frame=100 fps=30")
        buf.write_line("[error] connection refused")
        buf.write_line("frame=101 fps=30")
        buf.write_line("fatal: stream ended")
        errors = buf.get_error_lines()
        assert len(errors) == 2
        assert "[error] connection refused" in errors
        assert "fatal: stream ended" in errors

    def test_error_keywords_case_insensitive(self):
        buf = StderrRingBuffer()
        buf.write_line("ERROR: something")
        buf.write_line("Fatal crash")
        buf.write_line("INVALID input")
        buf.write_line("Operation FAILED")
        buf.write_line("ABORT signal")
        assert len(buf.get_error_lines()) == 5

    def test_no_errors_returns_empty(self):
        buf = StderrRingBuffer()
        buf.write_line("everything is fine")
        assert buf.get_error_lines() == []


class TestStderrRingBufferThreadSafety:
    """Concurrent access from multiple threads."""

    def test_concurrent_writes(self):
        buf = StderrRingBuffer(max_lines=1000, max_bytes=256 * 1024)
        num_threads = 4
        lines_per_thread = 200
        barrier = threading.Barrier(num_threads)

        def writer(thread_id):
            barrier.wait()
            for i in range(lines_per_thread):
                buf.write_line(f"t{thread_id}-{i}")

        threads = [
            threading.Thread(target=writer, args=(t,))
            for t in range(num_threads)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All lines should be present (800 < 1000 limit)
        assert buf.line_count == num_threads * lines_per_thread

    def test_concurrent_write_and_read(self):
        buf = StderrRingBuffer(max_lines=500, max_bytes=256 * 1024)
        stop = threading.Event()
        read_results = []

        def writer():
            for i in range(1000):
                buf.write_line(f"line-{i}")

        def reader():
            while not stop.is_set():
                lines = buf.get_lines()
                read_results.append(len(lines))

        w = threading.Thread(target=writer)
        r = threading.Thread(target=reader)
        r.start()
        w.start()
        w.join()
        stop.set()
        r.join()

        # Buffer should have at most 500 lines due to limit
        assert buf.line_count <= 500
        # Reader should have gotten results without errors
        assert len(read_results) > 0


class TestStderrRingBufferIdempotency:
    """Idempotency: repeated operations produce consistent results."""

    def test_double_clear_is_safe(self):
        buf = StderrRingBuffer()
        buf.write_line("data")
        buf.clear()
        buf.clear()
        assert buf.line_count == 0
        assert buf.total_bytes == 0

    def test_read_empty_buffer_is_safe(self):
        buf = StderrRingBuffer()
        assert buf.get_lines() == []
        assert buf.get_text() == ""
        assert buf.get_last_line() == ""
        assert buf.get_error_lines() == []
