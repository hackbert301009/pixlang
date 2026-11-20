"""Who may run a pipeline right now, and where a run's output goes."""
from __future__ import annotations

import contextlib
import sys
import threading
from typing import Any, Iterator, List, Optional

# A worker stuck in a native call cannot be killed from Python. Up to this many
# are abandoned so the editor stays usable; past it the process needs a restart.
MAX_STUCK_RUNS = 3
CANCEL_GRACE_S = 1.0     # how long a cancelled worker gets to unwind at its next statement
MAX_STDOUT_CHARS = 256 * 1024


class RunBlocked(RuntimeError):
    """Earlier runs are wedged and no new one is accepted."""


class RunTicket:
    def __init__(self, gate: RunGate) -> None:
        self._gate = gate
        self._held = True
        self._mutex = threading.Lock()

    def release(self) -> None:
        """Give the gate back; safe to call again after :meth:`abandon`."""
        with self._mutex:
            if not self._held:
                return
            self._held = False
        self._gate._lock.release()

    def abandon(self, worker: threading.Thread) -> None:
        """Stop waiting for *worker*: free the gate and remember the thread."""
        self._gate._stuck.append(worker)
        self.release()


class RunGate:
    """One pipeline at a time.

    A worker that outlives its deadline holds nothing the next run needs (runs
    share no process state), so its ticket may be abandoned instead of
    leaving every later run waiting on a lock nobody will release.
    """

    def __init__(self, max_stuck: int = MAX_STUCK_RUNS, grace: float = CANCEL_GRACE_S) -> None:
        self.grace = grace
        self._lock = threading.Lock()
        self._stuck: List[threading.Thread] = []
        self._max_stuck = max_stuck

    @property
    def stuck(self) -> int:
        self._stuck = [t for t in self._stuck if t.is_alive()]
        return len(self._stuck)

    def acquire(self, timeout: float) -> Optional[RunTicket]:
        """A ticket, or ``None`` when another run is still going."""
        if self.stuck >= self._max_stuck:
            raise RunBlocked(
                f"{self.stuck} earlier runs are stuck in a native call and cannot be stopped; "
                "restart the editor to clear them")
        if not self._lock.acquire(timeout=timeout):
            return None
        return RunTicket(self)


class ThreadStdout:
    """``sys.stdout`` stand-in: the owning thread's output goes to *buf*,
    everything else still reaches the real stream."""

    def __init__(self, buf: Any, tid: int, fallback: Any, limit: int = MAX_STDOUT_CHARS) -> None:
        self._buf, self._tid, self._fallback, self._limit = buf, tid, fallback, limit

    def write(self, s: str) -> int:
        if threading.get_ident() == self._tid:
            if self._buf.tell() < self._limit:
                self._buf.write(s)
            return len(s)
        return self._fallback.write(s)

    def flush(self) -> None:
        with contextlib.suppress(OSError, ValueError):
            self._fallback.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._fallback, name)


@contextlib.contextmanager
def capture_thread_stdout(buf: Any) -> Iterator[None]:
    """Capture what the calling thread prints without silencing other threads
    (``contextlib.redirect_stdout`` swallows everyone's output)."""
    previous = sys.stdout
    proxy = ThreadStdout(buf, threading.get_ident(), previous)
    sys.stdout = proxy  # type: ignore[assignment]
    try:
        yield
    finally:
        if sys.stdout is proxy:
            sys.stdout = previous
