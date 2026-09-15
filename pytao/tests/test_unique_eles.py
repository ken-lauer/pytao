import pytest

from .. import AnyTao
from ..errors import TaoCommandError
from .test_interface_commands import new_tao


def test_unique_eles_superuniverse(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        assert tao.unique_ele_ids() == ["1@0>>0", "1@0>>1", "1@0>>2"]


def test_unique_eles_universe(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        assert tao.unique_ele_ids("1") == ["1@0>>0", "1@0>>1", "1@0>>2"]


def test_unique_eles_branch(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        assert tao.unique_ele_ids("1@0") == ["1@0>>0", "1@0>>1", "1@0>>2"]


def test_unique_eles_element(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        assert tao.unique_ele_ids("1@0>>0") == ["1@0>>0"]


def test_unique_eles_missing(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/regression_tests/pipe_test/tao.init_wall"
    ) as tao:
        with pytest.raises(TaoCommandError):
            # this doesn't make it to sending a command -> inum must be int
            tao.unique_ele_ids("foo")


def test_eles_track_only(tao_cls: type[AnyTao]):
    with new_tao(
        tao_cls, init_file="$ACC_ROOT_DIR/bmad-doc/tao_examples/cbeta_cell/tao.init"
    ) as tao:
        default_names = {ele.head.name for ele in tao.eles("*", defaults=False)}
        tracked_names = {
            ele.head.name for ele in tao.eles("*", track_only=True, defaults=False)
        }

        assert "FF.QUA01" in default_names  # super lord
        assert "FF.QUA01" not in tracked_names
        assert "FF.QUA01#1" in tracked_names  # its super slave

        # The no-ele_id (superuniverse) path respects track_only, too
        assert {
            ele.head.name for ele in tao.eles(track_only=True, defaults=False)
        } == tracked_names
