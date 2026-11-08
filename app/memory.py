"""Keep the server's memory close to what it actually uses.

CPython hands freed memory back to glibc's malloc, which keeps it for reuse instead of
returning it to the system. Analyses run on worker threads, and glibc gives each thread
its own heap ("arena", up to eight per CPU the machine reports), so every big run leaves
a heap's worth of free memory behind that only that thread can reuse. On the 512 MB
server this kept ~400 MB in use while idle, and the next analysis went past the limit
(2026-11-08). Two arenas, and handing free memory back after each big run, keep the
idle server near what its caches actually hold.

Both are glibc calls; elsewhere (macOS, musl) they do nothing.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import logging
from collections.abc import Callable

M_ARENA_MAX = -8  # mallopt option, from glibc's malloc.h
ARENAS = 2

log = logging.getLogger(__name__)


def _libc() -> ctypes.CDLL | None:
    try:
        libc = ctypes.CDLL(ctypes.util.find_library("c") or "libc.so.6")
        libc.mallopt  # noqa: B018 - glibc only
        libc.malloc_trim  # noqa: B018
    except (OSError, AttributeError):
        return None
    return libc


_LIBC = _libc()


def limit_arenas(arenas: int = ARENAS) -> bool:
    """Cap glibc's per-thread heaps. Call before worker threads start; True if it took."""
    if _LIBC is None:
        return False
    ok = bool(_LIBC.mallopt(M_ARENA_MAX, arenas))
    if not ok:
        log.warning("mallopt(M_ARENA_MAX) failed")
    return ok


def give_back() -> None:
    """Return the heap's free pages to the system, e.g. after an analysis finishes."""
    if _LIBC is not None:
        _LIBC.malloc_trim(0)


def give_back_after[T](fn: Callable[..., T], *args, **kwargs) -> T:
    """Run ``fn`` (a big analysis), then return its freed memory to the system."""
    try:
        return fn(*args, **kwargs)
    finally:
        give_back()
