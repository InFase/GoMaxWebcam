"""Shared test utilities."""

from collections import deque


class FrameLog:
    """Memory-bounded frame log: keeps first 10 + last 90 frames.

    Prevents unbounded memory growth when the pipeline pushes thousands
    of frames per second, while still allowing assertions on both early
    and recent frames.
    """
    __slots__ = ('_first', '_recent', '_count', '_first_cap')

    def __init__(self, first_cap=10, recent_cap=90):
        self._first = []
        self._recent = deque(maxlen=recent_cap)
        self._count = 0
        self._first_cap = first_cap

    def append(self, frame):
        self._count += 1
        if len(self._first) < self._first_cap:
            self._first.append(frame)
        else:
            self._recent.append(frame)

    def __len__(self):
        return len(self._first) + len(self._recent)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return (self._first + list(self._recent))[idx]
        total = len(self._first) + len(self._recent)
        if idx < 0:
            idx += total
        if idx < len(self._first):
            return self._first[idx]
        return list(self._recent)[idx - len(self._first)]

    def __iter__(self):
        yield from self._first
        yield from self._recent

    def __bool__(self):
        return self._count > 0
