from __future__ import annotations

import contextlib
import logging
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

logger = logging.getLogger(__name__)

T = TypeVar("T")
R = TypeVar("R")

_READ_CHUNK = 1 << 20


@dataclass
class _Child:
    pid: int
    index: int
    fd: int
    buf: bytearray = field(default_factory=bytearray)


def error_result_from_exception(ex: BaseException) -> dict[str, str]:
    """Describe an exception in the same shape as `SubprocessErrorResult`."""
    return {
        "error": str(ex),
        "error_cls": type(ex).__name__,
        "traceback": traceback.format_exc(),
        "tao_output": getattr(ex, "tao_output", ""),
    }


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _child_main(
    tao: Tao,
    func: Callable,
    item: Any,
    write_fd: int,
    inherited_fds: Iterable[int],
) -> NoReturn:
    code = 0
    try:
        # Holding siblings' read ends would keep a write to an orphaned
        # parent from failing with EPIPE.
        for fd in inherited_fds:
            with contextlib.suppress(OSError):
                os.close(fd)
        try:
            payload = pickle.dumps(
                {"result": func(tao, item)}, protocol=pickle.HIGHEST_PROTOCOL
            )
        except BaseException as ex:  # noqa: BLE001
            code = 1
            payload = pickle.dumps(
                error_result_from_exception(ex), protocol=pickle.HIGHEST_PROTOCOL
            )
        _write_all(write_fd, payload)
        os.close(write_fd)
    except BaseException:  # noqa: BLE001
        code = 2
    finally:
        with contextlib.suppress(Exception):
            sys.stdout.flush()
            sys.stderr.flush()
        # Skip atexit handlers and finalizers; they belong to the parent.
        os._exit(code)


def _decode_child_payload(
    child: _Child,
    payload: bytes,
    exit_code: int,
    raises: bool,
) -> Any:
    index = child.index
    err: TaoForkError | None = None

    if exit_code < 0:
        signum = -exit_code
        try:
            name = signal.Signals(signum).name
        except ValueError:
            name = "unknown signal"
        err = TaoForkError(
            f"Child process for item {index} was killed by signal {signum} ({name})",
            index=index,
            exit_code=exit_code,
        )
    elif not payload:
        err = TaoForkError(
            f"Child process for item {index} exited with code {exit_code} "
            f"without returning a result",
            index=index,
            exit_code=exit_code,
        )
    else:
        try:
            value = pickle.loads(payload)
        except Exception as ex:  # noqa: BLE001
            err = TaoForkError(
                f"Could not decode the result for item {index}: {ex}",
                index=index,
                exit_code=exit_code,
            )
        else:
            if isinstance(value, dict) and "result" in value:
                return value["result"]
            if isinstance(value, dict) and "error" in value:
                err = TaoForkError(
                    f"Item {index} raised {value.get('error_cls')}: {value.get('error')}",
                    tao_output=value.get("tao_output", ""),
                    index=index,
                    error_cls=value.get("error_cls", ""),
                    child_traceback=value.get("traceback", ""),
                    exit_code=exit_code,
                )
                if hasattr(err, "add_note") and callable(err.add_note):
                    err.add_note(f"Child process {err.child_traceback}")
            else:
                err = TaoForkError(
                    f"Unexpected result payload for item {index}: {type(value).__name__}",
                    index=index,
                    exit_code=exit_code,
                )

    if raises:
        raise err
    return err


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


def fork_map(
    tao: Tao,
    func: Callable[[Tao, T], R],
    items: Iterable[T],
    *,
    max_workers: int | None = None,
    timeout: float | None = None,
    raises: bool = True,
) -> list[R | TaoForkError]:
    """
    Run `func(tao, item)` for each item in a forked copy of this process.

    See `Tao.fork_map` for full documentation.
    """
    if not hasattr(os, "fork"):
        raise NotImplementedError("fork_map requires os.fork(), which is unavailable here")

    workers = resolve_max_workers(max_workers)
    items = list(items)
    n = len(items)
    if n == 0:
        return []

    workers = min(workers, n)
    results: list[Any] = [None] * n
    pending = iter(range(n))
    live: dict[int, _Child] = {}
    sel = selectors.DefaultSelector()
    deadline = None if timeout is None else time.monotonic() + timeout

    def spawn(index: int) -> None:
        # Flush so buffered output is not duplicated into the child.
        sys.stdout.flush()
        sys.stderr.flush()
        read_fd, write_fd = os.pipe()
        pid = os.fork()
        if pid == 0:
            _child_main(tao, func, items[index], write_fd, [read_fd, *live])
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
                results[child.index] = _decode_child_payload(
                    child, bytes(child.buf), exit_code, raises
                )
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
            with contextlib.suppress(Exception):
                sel.unregister(child.fd)
            os.close(child.fd)
        live.clear()
        sel.close()
