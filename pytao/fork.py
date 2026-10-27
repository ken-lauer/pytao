from __future__ import annotations

import contextlib
import os
import pickle
import selectors
import signal
import sys
import time
import traceback
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, NoReturn, TypeVar

from .errors import TaoForkError

if TYPE_CHECKING:
    from .tao import Tao

T = TypeVar("T")
R = TypeVar("R")

_READ_CHUNK = 1 << 20

# Child exit codes.  A successful child sends its pickled result; a child
# whose `func` raised sends a pickled `TaoForkError`.
_EXIT_OK = 0
_EXIT_RAISED = 1


@dataclass
class _Child:
    pid: int
    index: int
    fd: int
    buf: bytearray = field(default_factory=bytearray)


def _child_main(
    tao: Tao,
    func: Callable,
    index: int,
    item: Any,
    write_fd: int,
    inherited_fds: Iterable[int],
) -> NoReturn:
    code = _EXIT_RAISED
    try:
        # Holding siblings' read ends would keep a write to an orphaned
        # parent from failing with EPIPE.
        for fd in inherited_fds:
            with contextlib.suppress(OSError):
                os.close(fd)
        try:
            payload = pickle.dumps(func(tao, item), protocol=pickle.HIGHEST_PROTOCOL)
            code = _EXIT_OK
        except BaseException as ex:  # noqa: BLE001
            err = TaoForkError(
                f"Item {index} raised {type(ex).__name__}: {ex}",
                tao_output=getattr(ex, "tao_output", ""),
                index=index,
                error_cls=type(ex).__name__,
                child_traceback=traceback.format_exc(),
                exit_code=_EXIT_RAISED,
            )
            if hasattr(err, "add_note"):
                err.add_note(f"Child process {err.child_traceback}")
            payload = pickle.dumps(err, protocol=pickle.HIGHEST_PROTOCOL)
        # The parent waits on this process once the pipe closes, so flush
        # (which can block on a full stdout pipe) must come first.
        with contextlib.suppress(Exception):
            sys.stdout.flush()
            sys.stderr.flush()
        with open(write_fd, "wb") as fp:
            fp.write(payload)
    finally:
        # Skip atexit handlers and finalizers; they belong to the parent.
        os._exit(code)


def _decode_payload(index: int, payload: bytes, exit_code: int) -> Any:
    """Unpickle a child's result, raising `TaoForkError` if it failed."""
    if exit_code < 0:
        try:
            name = signal.Signals(-exit_code).name
        except ValueError:
            name = "unknown signal"
        raise TaoForkError(
            f"Child process for item {index} was killed by signal {-exit_code} ({name})",
            index=index,
            exit_code=exit_code,
        )
    if exit_code not in (_EXIT_OK, _EXIT_RAISED) or not payload:
        raise TaoForkError(
            f"Child process for item {index} exited with code {exit_code} "
            f"without returning a result",
            index=index,
            exit_code=exit_code,
        )
    try:
        value = pickle.loads(payload)
    except Exception as ex:
        raise TaoForkError(
            f"Could not decode the result for item {index}: {ex}",
            index=index,
            exit_code=exit_code,
        ) from ex
    if exit_code == _EXIT_RAISED:
        raise value
    return value


def resolve_max_workers(max_workers: int | None) -> int:
    """
    Resolve a `fork_map` worker count to a positive number of processes.

    `None` means the number of CPUs; a negative value counts back from it, so
    `-1` leaves one CPU free.  The result is at least 1.
    """
    cpus = os.cpu_count() or 1
    if max_workers is None:
        return cpus
    if max_workers == 0:
        raise ValueError("max_workers must be a nonzero integer or None")
    if max_workers < 0:
        max_workers += cpus
    return max(1, max_workers)


def decode_fork_payloads(
    payloads: list[tuple[bytes, int]],
    raises: bool = True,
) -> list[Any]:
    """Decode the `(payload, exit_code)` pairs from `fork_map_payloads`."""
    results = []
    for index, (payload, exit_code) in enumerate(payloads):
        try:
            results.append(_decode_payload(index, payload, exit_code))
        except TaoForkError as err:
            if raises:
                raise
            results.append(err)
    return results


def fork_map_payloads(
    tao: Tao,
    func: Callable[[Tao, T], R],
    items: Iterable[T],
    *,
    max_workers: int | None = None,
    timeout: float | None = None,
    raises: bool = True,
) -> list[tuple[bytes, int]]:
    """
    Run `Tao.fork_map`, leaving each child's pickled result undecoded.

    Returns `(payload, exit_code)` pairs for `decode_fork_payloads`, letting a
    caller forward them without unpickling.  With `raises`, a failed child is
    decoded as soon as it finishes so that the first failure stops the rest.
    """
    if not hasattr(os, "fork"):
        raise NotImplementedError("fork_map requires os.fork(), which is unavailable here")

    workers = resolve_max_workers(max_workers)
    items = list(items)
    n = len(items)
    if n == 0:
        return []

    workers = min(workers, n)
    results: list[tuple[bytes, int]] = [(b"", 0)] * n
    pending = iter(range(n))
    live: dict[int, _Child] = {}
    sel = selectors.DefaultSelector()
    deadline = None if timeout is None else time.monotonic() + timeout

    def spawn(index: int) -> None:
        # Flush so buffered output is not duplicated into the child.
        sys.stdout.flush()
        sys.stderr.flush()
        read_fd, write_fd = os.pipe()
        try:
            pid = os.fork()
        except BaseException:
            os.close(read_fd)
            os.close(write_fd)
            raise
        if pid == 0:
            _child_main(tao, func, index, items[index], write_fd, [read_fd, *live])
        # The parent must drop its write end or EOF never arrives.
        os.close(write_fd)
        child = _Child(pid=pid, index=index, fd=read_fd)
        live[read_fd] = child
        sel.register(read_fd, selectors.EVENT_READ, child)

    try:
        for _ in range(workers):
            spawn(next(pending))

        while live:
            remaining = None
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(
                        f"fork_map timed out after {timeout} s with {len(live)} "
                        f"item(s) still running"
                    )

            for key, _ in sel.select(timeout=remaining):
                child: _Child = key.data
                chunk = os.read(key.fd, _READ_CHUNK)
                if chunk:
                    child.buf += chunk
                    continue

                sel.unregister(key.fd)
                os.close(key.fd)
                del live[key.fd]
                _, status = os.waitpid(child.pid, 0)
                exit_code = os.waitstatus_to_exitcode(status)
                payload = bytes(child.buf)
                if raises and exit_code != _EXIT_OK:
                    _decode_payload(child.index, payload, exit_code)
                results[child.index] = (payload, exit_code)
                next_index = next(pending, None)
                if next_index is not None:
                    spawn(next_index)

        return results
    finally:
        for child in live.values():
            with contextlib.suppress(ProcessLookupError):
                os.kill(child.pid, signal.SIGKILL)
        for child in live.values():
            with contextlib.suppress(ChildProcessError):
                os.waitpid(child.pid, 0)
            os.close(child.fd)
        sel.close()
