import pytest

from ..plotting.ele_methods import (
    CATEGORY_PALETTE,
    GARBAGE_COLOR,
    KNOWN_METHOD_COLORS,
    OFF_COLOR,
    color_for_value,
    is_garbage_value,
)
from .conftest import get_example, test_artifacts


@pytest.mark.parametrize("value", ["Off", "off", "OFF"])
def test_color_off_is_black(value: str):
    assert color_for_value(value) == OFF_COLOR == "#000000"


@pytest.mark.parametrize("value", ["GARBAGE!", "Garbage!", "garbage!"])
def test_color_garbage(value: str):
    assert color_for_value(value) == GARBAGE_COLOR
    assert is_garbage_value(value)


def test_known_colors_are_unique():
    colors = list(KNOWN_METHOD_COLORS.values())
    assert len(set(colors)) == len(colors)


def test_known_colors_are_case_insensitive():
    for name, color in KNOWN_METHOD_COLORS.items():
        assert color_for_value(name.upper()) == color


def test_unknown_value_color_is_deterministic():
    color = color_for_value("Some_Future_Method")
    assert color in CATEGORY_PALETTE
    assert color_for_value("some_future_method") == color


def test_plot_ele_methods():
    example = get_example("cbeta_cell")
    example.plot = "mpl"
    with example.run_context(use_subprocess=True) as tao:
        tao.plot_ele_methods(save=test_artifacts / "test_plot_ele_methods-mpl")
        data, _fig, axes = tao.last_plot

        assert data.names
        assert "tracking_method" in data.methods
        assert all(len(values) == len(data.names) for values in data.methods.values())
        assert all(end > start for start, end in zip(data.s_start, data.s_end))
        assert not data.csr_on
        assert len(axes) == 2  # lanes + layout

        tao.plot_ele_methods("quad::*", include_layout=False)
        subset, _fig, sub_axes = tao.last_plot
        assert set(subset.names) < set(data.names)
        assert len(sub_axes) == 1

        tao.plot_ele_methods(include_zero_length=True, include_layout=False)
        with_zero = tao.last_plot[0]
        assert len(with_zero.names) > len(data.names)
        assert any(start == end for start, end in zip(with_zero.s_start, with_zero.s_end))

        with pytest.raises(ValueError, match="not_a_method"):
            tao.plot_ele_methods(columns=["not_a_method"])


def test_plot_ele_methods_csr():
    example = get_example("csr_beam_tracking")
    example.plot = "mpl"
    with example.run_context(use_subprocess=True) as tao:
        tao.plot_ele_methods(save=test_artifacts / "test_plot_ele_methods-csr")
        data, _fig, axes = tao.last_plot

        assert data.csr_on
        assert data.n_bin is not None
        assert data.space_charge_mesh_size is not None
        assert len(axes) == 3  # lanes + csr_ds_step + layout

        tao.plot_ele_methods(show_csr_ds_step=False, include_layout=False)
        assert len(tao.last_plot[2]) == 1


def test_plot_ele_methods_bokeh_not_implemented():
    example = get_example("cbeta_cell")
    example.plot = "bokeh"
    with (
        example.run_context(use_subprocess=True) as tao,
        pytest.raises(NotImplementedError),
    ):
        tao.plot_ele_methods()
