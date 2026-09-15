from __future__ import annotations

import typing
import zlib

from pydantic import dataclasses

if typing.TYPE_CHECKING:
    from .. import Tao
    from ..model.ele import Which


#: Categorical `ele:methods` columns, in canonical display order.
METHOD_COLUMNS: tuple[str, ...] = (
    "tracking_method",
    "mat6_calc_method",
    "spin_tracking_method",
    "csr_method",
    "space_charge_method",
    "field_calc",
    "ptc_integration_type",
)

OFF_COLOR = "#000000"
GARBAGE_COLOR = "#ff0000"

#: Palette for categorical method values.  Black and red hues are excluded, as
#: they are reserved for "Off" and "GARBAGE!" respectively.
CATEGORY_PALETTE: tuple[str, ...] = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#c5b0d5",
    "#c49c94",
    "#f7b6d2",
    "#c7c7c7",
    "#dbdb8d",
    "#9edae5",
    "#393b79",
    "#5254a3",
    "#637939",
    "#8ca252",
    "#8c6d31",
    "#bd9e39",
    "#7b4173",
    "#a55194",
    "#ce6dbd",
    "#9c9ede",
    "#cedb9c",
    "#e7ba52",
)

#: Dedicated colors for all known method names from bmad_struct.f90's
#: `*_method_name`-style arrays, so that colors are consistent between runs.
#: Colors are hand-assigned for contrast between values commonly displayed
#: together (e.g., Bmad_Standard / Auto / Tracking / 1_Dim / Matrix_Kick).
KNOWN_METHOD_COLORS: dict[str, str] = {
    "Off": OFF_COLOR,
    "GARBAGE!": GARBAGE_COLOR,
    # tracking_method_name
    "Bmad_Standard": "#1f77b4",
    "Symp_Lie_PTC": "#ff7f0e",
    "Runge_Kutta": "#2ca02c",
    "Linear": "#9467bd",
    "Time_Runge_Kutta": "#98df8a",
    "Custom": "#8c564b",
    "Taylor": "#e377c2",
    "Fixed_Step_Runge_Kutta": "#637939",
    "Symp_Lie_Bmad": "#17becf",
    "Fixed_Step_Time_Runge_kutta": "#9edae5",
    "MAD": "#bd9e39",
    # spin_tracking_method_name
    "Transverse_Kick": "#ffbb78",
    "Tracking": "#c5b0d5",
    "Magnus": "#7b4173",
    "Sprint": "#aec7e8",
    # mat6_calc_method_name
    "Auto": "#7f7f7f",
    # csr_method_name
    "1_Dim": "#bcbd22",
    "Steady_State_3D": "#393b79",
    # space_charge_method_name
    "Slice": "#e7ba52",
    "FFT_3D": "#5254a3",
    "Cathode_FFT_3D": "#ce6dbd",
    # field_calc_name
    "FieldMap": "#8ca252",
    "Planar_Model": "#c49c94",
    "Refer_to_Lords.": "#c7c7c7",
    "No_Field": "#f7b6d2",
    "Helical_Model": "#a55194",
    "Soft_edge": "#dbdb8d",
    # ptc_integration_type_name
    "Drift_Kick": "#cedb9c",
    "Matrix_Kick": "#9c9ede",
    "Ripken_Kick": "#8c6d31",
}

METHOD_COLORS: dict[str, str] = {
    name.lower(): color for name, color in KNOWN_METHOD_COLORS.items()
}


def is_garbage_value(value: str) -> bool:
    """A "GARBAGE!" method value indicates an issue with bmad itself."""
    return value.lower() == "garbage!"


def color_for_value(value: str) -> str:
    """
    Get the display color (as a hex string) for a categorical method value.

    Known values have dedicated colors; unknown ones get a deterministic
    fallback so that colors remain consistent between runs.
    """
    key = value.lower()
    try:
        return METHOD_COLORS[key]
    except KeyError:
        return CATEGORY_PALETTE[zlib.crc32(key.encode()) % len(CATEGORY_PALETTE)]


def _is_active(value: str | None) -> bool:
    return value is not None and value.lower() != "off"


@dataclasses.dataclass
class ElementMethodsPlotData:
    """
    Per-element method settings gathered for `plot_ele_methods`.

    All per-element lists share the same length and ordering (by increasing
    longitudinal position).
    """

    ix_eles: list[int]
    names: list[str]
    s_start: list[float]
    s_end: list[float]
    methods: dict[str, list[str | None]]
    csr_ds_step: list[float | None]
    space_charge_mesh_size: list[int] | None = None
    csr3d_mesh_size: list[int] | None = None
    n_bin: int | None = None

    @property
    def csr_on(self) -> bool:
        """CSR is active (not "Off") for at least one element."""
        return any(_is_active(value) for value in self.methods.get("csr_method", []))

    @property
    def csr_3d_on(self) -> bool:
        """3D CSR (Steady_State_3D) is active for at least one element."""
        return any(
            value is not None and value.lower() == "steady_state_3d"
            for value in self.methods.get("csr_method", [])
        )

    @property
    def space_charge_on(self) -> bool:
        """Space charge is active (not "Off") for at least one element."""
        return any(_is_active(value) for value in self.methods.get("space_charge_method", []))

    @classmethod
    def from_tao(
        cls,
        tao: Tao,
        ele_id: str = "*",
        *,
        ix_uni: str = "1",
        ix_branch: str = "0",
        which: Which = "model",
        include_zero_length: bool = False,
    ) -> ElementMethodsPlotData:
        """
        Gather element method settings from Tao.

        Parameters
        ----------
        tao : Tao
        ele_id : str, default="*"
            Element match string, using the same syntax as `Tao.eles`
            (e.g., ``"*"``, ``"1:20"``, ``"quad::*"``).
            Only tracking elements are considered; lord elements are excluded.
        ix_uni : str, default="1"
            Universe index.
        ix_branch : str, default="0"
            Branch index.
        which : "model", "base", or "design", default="model"
        include_zero_length : bool, default=False
            Include zero-length elements.
        """
        # Lord elements overlap their slaves longitudinally, which would draw
        # conflicting blocks on top of each other.
        elements = tao.eles(
            ele_id,
            ix_uni=ix_uni,
            ix_branch=ix_branch,
            which=which,
            track_only=True,
            defaults=False,
            attrs=True,
            methods=True,
        )

        ix_eles: list[int] = []
        names: list[str] = []
        s_start: list[float] = []
        s_end: list[float] = []
        csr_ds_step: list[float | None] = []
        methods: dict[str, list[str | None]] = {col: [] for col in METHOD_COLUMNS}

        for ele in sorted(elements, key=lambda ele: ele.head.s):
            length_attr = ele.attrs.attrs.get("L") if ele.attrs is not None else None
            length = float(length_attr.data) if length_attr is not None else 0.0
            if length <= 0.0 and not include_zero_length:
                continue

            ix_eles.append(ele.head.ix_ele)
            names.append(ele.head.name)
            s_start.append(ele.head.s_start)
            s_end.append(ele.head.s)

            ds_attr = ele.attrs.attrs.get("csr_ds_step") if ele.attrs is not None else None
            csr_ds_step.append(float(ds_attr.data) if ds_attr is not None else None)

            for col in METHOD_COLUMNS:
                methods[col].append(
                    getattr(ele.methods, col) if ele.methods is not None else None
                )

        methods = {
            col: values
            for col, values in methods.items()
            if any(value is not None for value in values)
        }

        data = cls(
            ix_eles=ix_eles,
            names=names,
            s_start=s_start,
            s_end=s_end,
            methods=methods,
            csr_ds_step=csr_ds_step,
        )

        if data.csr_on or data.space_charge_on:
            space_charge_com = tao.space_charge_com()
            data.space_charge_mesh_size = [
                int(v) for v in space_charge_com["space_charge_mesh_size"]
            ]
            data.csr3d_mesh_size = [int(v) for v in space_charge_com["csr3d_mesh_size"]]
            data.n_bin = int(space_charge_com["n_bin"])

        return data
