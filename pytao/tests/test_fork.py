from __future__ import annotations

import os
import pickle
import selectors
import signal
import time
from collections.abc import Generator
from typing import Any

import numpy as np
import pytest

from .. import SubprocessTao, Tao, TaoForkError
from ..subproc import AnyTao
from .conftest import new_tao

pytestmark = pytest.mark.filterwarnings(
    "ignore:This process.*multi-threaded:DeprecationWarning"
)

OPTICS_INIT = "-init $ACC_ROOT_DIR/bmad-doc/tao_examples/optics_matching/tao.init"
CSR_INIT = "-init $ACC_ROOT_DIR/regression_tests/pipe_test/csr_beam_tracking/tao.init"


@pytest.fixture
def tao(tao_cls: type[AnyTao]) -> Generator[AnyTao]:
    with new_tao(tao_cls, OPTICS_INIT) as tao:
        yield tao


@pytest.fixture
def plain_tao() -> Generator[Tao]:
    with new_tao(Tao, OPTICS_INIT) as tao:
        yield tao


@pytest.fixture
def forked_pids(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Record the pids of every child forked by this process during the test."""
    pids: list[int] = []
    orig_fork = os.fork

    def recording_fork() -> int:
        pid = orig_fork()
        if pid:
            pids.append(pid)
        return pid

    monkeypatch.setattr(os, "fork", recording_fork)
    return pids


def assert_all_reaped(pids: list[int]) -> None:
    for pid in pids:
        with pytest.raises(ChildProcessError):
            os.waitpid(pid, os.WNOHANG)


def set_k1_and_read(tao: Tao, k1: float) -> float:
    tao.cmd(f"set ele Q1 k1 = {k1}")
    return float(tao.ele_gen_attribs("Q1")["K1"])


def beta_for_k1(tao: Tao, k1: float) -> float:
    tao.cmd(f"set ele Q1 k1 = {k1}")
    return float(tao.ele_twiss("Q2")["beta_a"])


def sleep_then_return(tao: Tao, item: tuple[float, int]) -> int:
    delay, value = item
    time.sleep(delay)
    return value


def identity(tao: Tao, item: Any) -> Any:
    return item


def raise_on_two(tao: Tao, item: int) -> int:
    if item == 2:
        raise ValueError("boom")
    return item


def bad_tao_command(tao: Tao, item: Any) -> None:
    tao.cmd("set ele NOPE k1 = 1")


def exit_three(tao: Tao, item: Any) -> None:
    os._exit(3)


def kill_self(tao: Tao, item: Any) -> None:
    os.kill(os.getpid(), signal.SIGKILL)


def sleep_forever(tao: Tao, item: Any) -> None:
    time.sleep(30)


def bunch_x(tao: Tao, item: Any) -> np.ndarray:
    return tao.bunch1(ele_id="end", coordinate="x", which="model", ix_bunch="1")


def test_independent_results(tao: AnyTao) -> None:
    k1_before = float(tao.ele_gen_attribs("Q1")["K1"])
    items = [1.0, 2.0, 3.0]

    assert tao.fork_map(set_k1_and_read, items) == items
    assert float(tao.ele_gen_attribs("Q1")["K1"]) == k1_before


def test_results_differ_per_item(tao: AnyTao) -> None:
    betas = tao.fork_map(beta_for_k1, [0.0, 0.5, 1.0, 1.5], max_workers=2)
    assert len(set(betas)) == 4


def test_order_preserved(tao: AnyTao) -> None:
    items = [(0.3, 0), (0.2, 1), (0.1, 2)]
    assert tao.fork_map(sleep_then_return, items, max_workers=3) == [0, 1, 2]


@pytest.mark.parametrize("max_workers", [1, 2, 4, None])
def test_max_workers(tao: AnyTao, max_workers: int | None) -> None:
    assert tao.fork_map(identity, range(6), max_workers=max_workers) == list(range(6))


def test_empty_items(tao: AnyTao) -> None:
    assert tao.fork_map(identity, []) == []


def test_func_raises(tao: AnyTao) -> None:
    with pytest.raises(TaoForkError) as excinfo:
        tao.fork_map(raise_on_two, range(4))

    err = excinfo.value
    assert err.index == 2
    assert err.error_cls == "ValueError"
    assert "boom" in str(err)
    assert "ValueError: boom" in err.child_traceback


def test_func_raises_returned_in_place(tao: AnyTao) -> None:
    results = tao.fork_map(raise_on_two, range(4), raises=False)
    assert results[:2] == [0, 1]
    assert results[3] == 3
    assert isinstance(results[2], TaoForkError)
    assert results[2].index == 2


def test_tao_command_error_in_child(tao: AnyTao) -> None:
    with pytest.raises(TaoForkError) as excinfo:
        tao.fork_map(bad_tao_command, [0])

    assert excinfo.value.error_cls == "TaoCommandError"
    assert excinfo.value.tao_output


def test_child_exit_without_result(tao: AnyTao) -> None:
    with pytest.raises(TaoForkError, match="exited with code 3") as excinfo:
        tao.fork_map(exit_three, [0])
    assert excinfo.value.exit_code == 3


def test_child_killed_by_signal(tao: AnyTao) -> None:
    with pytest.raises(TaoForkError, match="SIGKILL") as excinfo:
        tao.fork_map(kill_self, [0])
    assert excinfo.value.exit_code == -signal.SIGKILL


def test_timeout(tao: AnyTao) -> None:
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        tao.fork_map(sleep_forever, range(3), timeout=0.5)
    assert time.monotonic() - t0 < 5.0

    assert tao.ele_twiss("Q2")["beta_a"] > 0


def test_timeout_reaps_children(plain_tao: Tao, forked_pids: list[int]) -> None:
    with pytest.raises(TimeoutError):
        plain_tao.fork_map(sleep_forever, range(3), timeout=0.5)

    assert len(forked_pids) == 3
    assert_all_reaped(forked_pids)


def test_failure_kills_remaining_children(plain_tao: Tao, forked_pids: list[int]) -> None:
    def fail_fast_or_sleep(tao: Tao, item: int) -> int:
        if item == 0:
            raise RuntimeError("first")
        time.sleep(30)
        return item

    t0 = time.monotonic()
    with pytest.raises(TaoForkError):
        plain_tao.fork_map(fail_fast_or_sleep, range(3), max_workers=3)
    assert time.monotonic() - t0 < 5.0
    assert_all_reaped(forked_pids)


def test_keyboard_interrupt_reaps_children(
    plain_tao: Tao,
    forked_pids: list[int],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def interrupt(self, timeout=None):
        raise KeyboardInterrupt

    monkeypatch.setattr(selectors.DefaultSelector, "select", interrupt)
    with pytest.raises(KeyboardInterrupt):
        plain_tao.fork_map(sleep_forever, range(2), max_workers=2)

    assert len(forked_pids) == 2
    assert_all_reaped(forked_pids)


def test_closure_and_lambda_in_process(plain_tao: Tao) -> None:
    offset = 10

    def add_offset(tao: Tao, item: int) -> int:
        return item + offset

    assert plain_tao.fork_map(add_offset, [1, 2]) == [11, 12]
    assert plain_tao.fork_map(lambda tao, item: item * 2, [1, 2]) == [2, 4]


def test_closure_and_lambda_in_subprocess() -> None:
    pytest.importorskip("cloudpickle")
    offset = 10

    def add_offset(tao: Tao, item: int) -> int:
        return item + offset

    with new_tao(SubprocessTao, OPTICS_INIT) as tao:
        assert tao.fork_map(add_offset, [1, 2]) == [11, 12]
        assert tao.fork_map(lambda tao, item: item * 2, [1, 2]) == [2, 4]


def test_subprocess_without_cloudpickle(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("pytao.subproc.cloudpickle", None)

    def local_func(tao: Tao, item: int) -> int:
        return item

    with new_tao(SubprocessTao, OPTICS_INIT) as tao:
        with pytest.raises(ValueError, match="cloudpickle"):
            tao.fork_map(local_func, [1])
        with pytest.raises(ValueError, match="cloudpickle"):
            tao.fork_map(lambda tao, item: item, [1])
        assert tao.fork_map(identity, [1, 2]) == [1, 2]


def test_numpy_result_round_trip(tao_cls: type[AnyTao]) -> None:
    with new_tao(tao_cls, CSR_INIT) as tao:
        expected = bunch_x(tao, None)
        (result,) = tao.fork_map(bunch_x, [None])
    assert isinstance(result, np.ndarray)
    assert np.allclose(result, expected)


def test_fork_error_pickle_round_trip() -> None:
    err = TaoForkError(
        "msg",
        tao_output="out",
        index=3,
        error_cls="ValueError",
        child_traceback="tb",
        exit_code=-9,
    )
    restored = pickle.loads(pickle.dumps(err))
    assert isinstance(restored, TaoForkError)
    assert str(restored) == "msg"
    assert restored.tao_output == "out"
    assert restored.index == 3
    assert restored.error_cls == "ValueError"
    assert restored.child_traceback == "tb"
    assert restored.exit_code == -9


def test_not_implemented_without_fork(plain_tao: Tao, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "fork")
    with pytest.raises(NotImplementedError):
        plain_tao.fork_map(identity, [1])
