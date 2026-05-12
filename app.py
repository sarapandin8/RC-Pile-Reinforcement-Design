from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Literal

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


Shape = Literal["Square", "Rectangle", "Circular"]

DEFAULT_BAR_SIZES_MM = [12.0, 16.0, 20.0, 25.0, 28.0, 32.0]


@dataclass(frozen=True)
class SectionGeometry:
    shape: Shape
    ag_mm2: float
    width_mm: float | None = None
    depth_mm: float | None = None
    diameter_mm: float | None = None


@dataclass(frozen=True)
class DesignResult:
    ast_required_mm2: float
    ast_axial_starter_mm2: float
    ast_from_equation_mm2: float
    ast_min_mm2: float
    ast_max_mm2: float
    phi_pn_kN: float
    steel_ratio_percent: float
    spacing_ok: bool
    spacing_note: str
    overall_ok: bool
    minimum_eccentricity_mm: float
    governing_ratio: float = 0.0
    governing_load_case: str = ""
    phi_mnx_at_pu_kNm: float = 0.0
    phi_mny_at_pu_kNm: float = 0.0
    pmm_required_found: bool = False


@dataclass(frozen=True)
class InteractionPoint:
    phi_pn_kN: float
    phi_mn_kNm: float
    neutral_axis_mm: float
    max_tension_strain: float


def bar_area_mm2(diameter_mm: float) -> float:
    return math.pi * diameter_mm**2 / 4.0


def phi_factor(column_type: str) -> float:
    return 0.65 if column_type == "Tied" else 0.75


def strain_based_phi(column_type: str, fy_mpa: float, tensile_strain: float, es_mpa: float = 200000.0) -> float:
    phi_compression = phi_factor(column_type)
    phi_tension = 0.90
    epsilon_y = fy_mpa / es_mpa
    if tensile_strain <= epsilon_y:
        return phi_compression
    if tensile_strain >= 0.005:
        return phi_tension
    transition = (tensile_strain - epsilon_y) / (0.005 - epsilon_y) if 0.005 > epsilon_y else 1.0
    transition = max(0.0, min(1.0, transition))
    return phi_compression + transition * (phi_tension - phi_compression)


def section_geometry(shape: Shape, width_mm: float | None, depth_mm: float | None, diameter_mm: float | None) -> SectionGeometry:
    if shape == "Square":
        ag = float(width_mm) * float(width_mm)
        return SectionGeometry(shape=shape, ag_mm2=ag, width_mm=float(width_mm), depth_mm=float(width_mm))
    if shape == "Rectangle":
        ag = float(width_mm) * float(depth_mm)
        return SectionGeometry(shape=shape, ag_mm2=ag, width_mm=float(width_mm), depth_mm=float(depth_mm))
    ag = math.pi * float(diameter_mm) ** 2 / 4.0
    return SectionGeometry(shape=shape, ag_mm2=ag, diameter_mm=float(diameter_mm))


def effective_cover_mm(clear_cover_mm: float, tie_dia_mm: float, main_bar_dia_mm: float) -> float:
    return clear_cover_mm + tie_dia_mm + 0.5 * main_bar_dia_mm


def axial_cap_factor(column_type: str) -> float:
    return 0.80 if column_type == "Tied" else 0.85


def nominal_axial_strength_n(fc_mpa: float, fy_mpa: float, ag_mm2: float, ast_mm2: float) -> float:
    return 0.85 * fc_mpa * (ag_mm2 - ast_mm2) + fy_mpa * ast_mm2


def design_strength_kN(
    phi: float,
    fc_mpa: float,
    fy_mpa: float,
    ag_mm2: float,
    ast_mm2: float,
    axial_cap_factor_value: float = 1.0,
) -> float:
    nominal_n = nominal_axial_strength_n(fc_mpa, fy_mpa, ag_mm2, ast_mm2)
    return axial_cap_factor_value * phi * nominal_n / 1000.0


def default_load_cases() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"Case": "LC1", "Pu (kN)": 1200.0, "Mx (kN-m)": 0.0, "My (kN-m)": 0.0},
            {"Case": "LC2", "Pu (kN)": 1000.0, "Mx (kN-m)": 60.0, "My (kN-m)": 40.0},
            {"Case": "LC3", "Pu (kN)": 900.0, "Mx (kN-m)": 90.0, "My (kN-m)": 55.0},
        ]
    )


def required_steel_area_mm2(
    pu_kN: float,
    phi: float,
    fc_mpa: float,
    fy_mpa: float,
    ag_mm2: float,
    min_ratio: float,
    axial_cap_factor_value: float = 1.0,
) -> tuple[float, float, float]:
    pu_n = pu_kN * 1000.0
    ast_min = min_ratio * ag_mm2
    denominator_strength = axial_cap_factor_value * phi
    numerator = pu_n / denominator_strength - 0.85 * fc_mpa * ag_mm2 if denominator_strength > 0 else float("inf")
    denominator = fy_mpa - 0.85 * fc_mpa
    ast_equation = numerator / denominator if denominator > 0 else 0.0
    ast_required = max(ast_min, ast_equation, 0.0)
    return ast_required, ast_equation, ast_min


def minimum_eccentricity_mm(geometry: SectionGeometry) -> float:
    if geometry.shape == "Circular":
        return 0.1 * float(geometry.diameter_mm)
    return 0.1 * min(float(geometry.width_mm), float(geometry.depth_mm))


def minimum_moment_kNm(pu_kN: float, eccentricity_mm: float) -> float:
    if pu_kN <= 0.0 or eccentricity_mm <= 0.0:
        return 0.0
    return pu_kN * eccentricity_mm / 1000.0


def enforce_minimum_eccentricity(
    pu_kN: float,
    mx_kNm: float,
    my_kNm: float,
    eccentricity_mm: float,
    mnx_kNm: float,
    mny_kNm: float,
) -> tuple[float, float, float, bool]:
    m_min_kNm = minimum_moment_kNm(pu_kN, eccentricity_mm)
    resultant = math.hypot(mx_kNm, my_kNm)
    if m_min_kNm <= 0.0 or resultant >= m_min_kNm:
        return mx_kNm, my_kNm, m_min_kNm, False
    if resultant > 1e-9:
        scale = m_min_kNm / resultant
        return mx_kNm * scale, my_kNm * scale, m_min_kNm, True
    if mnx_kNm <= mny_kNm:
        return m_min_kNm, 0.0, m_min_kNm, True
    return 0.0, m_min_kNm, m_min_kNm, True


def layout_total_bars(
    shape: Shape,
    bars_width_face: int | None = None,
    bars_depth_face: int | None = None,
    bars_circular: int | None = None,
) -> int:
    if shape == "Circular":
        return int(bars_circular or 0)
    return total_bars_rectangular(int(bars_width_face or 0), int(bars_depth_face or 0))


def format_required_ast(ast_required_mm2: float, found: bool, ast_max_mm2: float) -> str:
    if found and math.isfinite(ast_required_mm2):
        return f"{ast_required_mm2:,.1f} mm2"
    return f"> {ast_max_mm2:,.1f} mm2"


def minimum_clear_spacing_mm(bar_dia_mm: float, user_min_clear_spacing_mm: float) -> float:
    return max(user_min_clear_spacing_mm, 1.5 * bar_dia_mm, 40.0)


def circular_spacing_check(
    diameter_mm: float,
    clear_cover_mm: float,
    tie_dia_mm: float,
    bar_dia_mm: float,
    n_bars: int,
    min_clear_spacing_mm: float,
) -> tuple[bool, str]:
    core_dia = diameter_mm - 2.0 * effective_cover_mm(clear_cover_mm, tie_dia_mm, bar_dia_mm)
    if core_dia <= 0:
        return False, "Cover, tie, and main bar consume the whole section."
    circumference = math.pi * core_dia
    center_spacing = circumference / n_bars
    clear_spacing = center_spacing - bar_dia_mm
    ok = clear_spacing >= min_clear_spacing_mm
    return ok, f"clear spacing = {clear_spacing:,.1f} mm"


def rectangular_spacing_check(
    width_mm: float,
    depth_mm: float,
    clear_cover_mm: float,
    tie_dia_mm: float,
    bar_dia_mm: float,
    bars_width_face: int,
    bars_depth_face: int,
    min_clear_spacing_mm: float,
) -> tuple[bool, str]:
    core_width = width_mm - 2.0 * effective_cover_mm(clear_cover_mm, tie_dia_mm, bar_dia_mm)
    core_depth = depth_mm - 2.0 * effective_cover_mm(clear_cover_mm, tie_dia_mm, bar_dia_mm)
    if core_width <= 0 or core_depth <= 0:
        return False, "Cover, tie, and main bar consume the whole section."

    width_clear = float("inf") if bars_width_face <= 1 else core_width / (bars_width_face - 1) - bar_dia_mm
    depth_clear = float("inf") if bars_depth_face <= 1 else core_depth / (bars_depth_face - 1) - bar_dia_mm
    ok = width_clear >= min_clear_spacing_mm and depth_clear >= min_clear_spacing_mm
    return ok, f"clear spacing x = {width_clear:,.1f} mm, y = {depth_clear:,.1f} mm"


def total_bars_rectangular(bars_width_face: int, bars_depth_face: int) -> int:
    if bars_width_face < 2 or bars_depth_face < 2:
        return 0
    return 2 * bars_width_face + 2 * max(0, bars_depth_face - 2)


def rectangular_bar_coordinates(
    width_mm: float,
    depth_mm: float,
    clear_cover_mm: float,
    tie_dia_mm: float,
    bar_dia_mm: float,
    bars_width_face: int,
    bars_depth_face: int,
) -> list[tuple[float, float]]:
    x_edge = width_mm / 2.0 - effective_cover_mm(clear_cover_mm, tie_dia_mm, bar_dia_mm)
    y_edge = depth_mm / 2.0 - effective_cover_mm(clear_cover_mm, tie_dia_mm, bar_dia_mm)
    xs = [0.0] if bars_width_face == 1 else [(-x_edge + i * (2.0 * x_edge / (bars_width_face - 1))) for i in range(bars_width_face)]
    ys = [0.0] if bars_depth_face == 1 else [(-y_edge + i * (2.0 * y_edge / (bars_depth_face - 1))) for i in range(bars_depth_face)]

    coords: list[tuple[float, float]] = []
    for x in xs:
        coords.append((x, y_edge))
        coords.append((x, -y_edge))
    for y in ys[1:-1]:
        coords.append((x_edge, y))
        coords.append((-x_edge, y))
    return coords


def circular_bar_coordinates(
    diameter_mm: float,
    clear_cover_mm: float,
    tie_dia_mm: float,
    bar_dia_mm: float,
    n_bars: int,
) -> list[tuple[float, float]]:
    radius = diameter_mm / 2.0 - effective_cover_mm(clear_cover_mm, tie_dia_mm, bar_dia_mm)
    return [
        (radius * math.cos(2.0 * math.pi * i / n_bars), radius * math.sin(2.0 * math.pi * i / n_bars))
        for i in range(n_bars)
    ]


def beta1_aci(fc_mpa: float) -> float:
    return max(0.65, min(0.85, 0.85 - 0.05 * max(fc_mpa - 28.0, 0.0) / 7.0))


def section_bar_coordinates(
    geometry: SectionGeometry,
    shape: Shape,
    cover_mm: float,
    tie_dia_mm: float,
    bar_dia_mm: float,
    bars_width_face: int | None = None,
    bars_depth_face: int | None = None,
    bars_circular: int | None = None,
) -> list[tuple[float, float]]:
    if shape == "Circular":
        return circular_bar_coordinates(float(geometry.diameter_mm), cover_mm, tie_dia_mm, bar_dia_mm, int(bars_circular or 0))
    return rectangular_bar_coordinates(
        float(geometry.width_mm),
        float(geometry.depth_mm),
        cover_mm,
        tie_dia_mm,
        bar_dia_mm,
        int(bars_width_face or 0),
        int(bars_depth_face or 0),
    )


def section_depth_along_axis(geometry: SectionGeometry, axis: Literal["x", "y"]) -> float:
    if geometry.shape == "Circular":
        return float(geometry.diameter_mm)
    return float(geometry.depth_mm) if axis == "x" else float(geometry.width_mm)


def section_width_at_coordinate(geometry: SectionGeometry, axis: Literal["x", "y"], coord_mm: float) -> float:
    if geometry.shape == "Circular":
        radius = float(geometry.diameter_mm) / 2.0
        if abs(coord_mm) >= radius:
            return 0.0
        return 2.0 * math.sqrt(radius**2 - coord_mm**2)
    if axis == "x":
        return float(geometry.width_mm)
    return float(geometry.depth_mm)


def steel_stress_mpa(strain: float, fy_mpa: float, es_mpa: float = 200000.0) -> float:
    return max(-fy_mpa, min(fy_mpa, es_mpa * strain))


def section_response(
    geometry: SectionGeometry,
    axis: Literal["x", "y"],
    neutral_axis_mm: float,
    fc_mpa: float,
    fy_mpa: float,
    bar_area_single_mm2: float,
    bar_coords: list[tuple[float, float]],
) -> tuple[float, float, float]:
    eps_cu = 0.003
    beta1 = beta1_aci(fc_mpa)
    depth = section_depth_along_axis(geometry, axis)
    half_depth = depth / 2.0
    a_depth = min(beta1 * neutral_axis_mm, depth)
    compression_depth = min(a_depth, depth)

    strip_count = 240
    strip_thickness = compression_depth / strip_count if compression_depth > 0 else 0.0
    concrete_force_n = 0.0
    concrete_moment_nmm = 0.0

    for i in range(strip_count):
        top_face_coord = half_depth - (i + 0.5) * strip_thickness
        strip_width = section_width_at_coordinate(geometry, axis, top_face_coord)
        strip_area = strip_width * strip_thickness
        strip_force = 0.85 * fc_mpa * strip_area
        concrete_force_n += strip_force
        concrete_moment_nmm += strip_force * top_face_coord

    steel_force_n = 0.0
    steel_moment_nmm = 0.0
    max_tension_strain = 0.0
    block_limit = half_depth - compression_depth

    for x, y in bar_coords:
        coord = y if axis == "x" else x
        dist_from_top = half_depth - coord
        strain = eps_cu * (1.0 - dist_from_top / neutral_axis_mm)
        max_tension_strain = max(max_tension_strain, -strain)
        stress = steel_stress_mpa(strain, fy_mpa)
        in_compression_block = coord >= block_limit
        steel_force = bar_area_single_mm2 * (stress - 0.85 * fc_mpa) if in_compression_block else bar_area_single_mm2 * stress
        steel_force_n += steel_force
        steel_moment_nmm += steel_force * coord

    nominal_p_n = concrete_force_n + steel_force_n
    nominal_m_nmm = concrete_moment_nmm + steel_moment_nmm
    return nominal_p_n, nominal_m_nmm, max_tension_strain


def interaction_curve(
    geometry: SectionGeometry,
    shape: Shape,
    cover_mm: float,
    tie_dia_mm: float,
    bar_dia_mm: float,
    fc_mpa: float,
    fy_mpa: float,
    phi: float,
    column_type: str,
    axial_cap_factor_value: float,
    axis: Literal["x", "y"],
    bar_area_single_override_mm2: float | None = None,
    bars_width_face: int | None = None,
    bars_depth_face: int | None = None,
    bars_circular: int | None = None,
) -> pd.DataFrame:
    coords = section_bar_coordinates(
        geometry=geometry,
        shape=shape,
        cover_mm=cover_mm,
        tie_dia_mm=tie_dia_mm,
        bar_dia_mm=bar_dia_mm,
        bars_width_face=bars_width_face,
        bars_depth_face=bars_depth_face,
        bars_circular=bars_circular,
    )
    bar_area_single = bar_area_single_override_mm2 if bar_area_single_override_mm2 is not None else bar_area_mm2(bar_dia_mm)
    total_ast_mm2 = len(coords) * bar_area_single
    depth = section_depth_along_axis(geometry, axis)
    c_values = [1.0, 5.0, 10.0]
    c_values.extend([depth * frac for frac in [0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.65, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0]])
    c_values.extend([depth + extra for extra in [100.0, 200.0, 400.0, 800.0, 1200.0, 2000.0, 4000.0]])
    unique_c = sorted({round(max(1.0, c), 6) for c in c_values})

    rows: list[InteractionPoint] = []
    for c in unique_c:
        nominal_p_n, nominal_m_nmm, max_tension_strain = section_response(
            geometry=geometry,
            axis=axis,
            neutral_axis_mm=c,
            fc_mpa=fc_mpa,
            fy_mpa=fy_mpa,
            bar_area_single_mm2=bar_area_single,
            bar_coords=coords,
        )
        rows.append(
            InteractionPoint(
                phi_pn_kN=strain_based_phi(column_type, fy_mpa, max_tension_strain) * nominal_p_n / 1000.0,
                phi_mn_kNm=strain_based_phi(column_type, fy_mpa, max_tension_strain) * abs(nominal_m_nmm) / 1_000_000.0,
                neutral_axis_mm=c,
                max_tension_strain=max_tension_strain,
            )
        )

    df = pd.DataFrame([point.__dict__ for point in rows])
    phi_pn_max_kN = design_strength_kN(
        phi=phi,
        fc_mpa=fc_mpa,
        fy_mpa=fy_mpa,
        ag_mm2=geometry.ag_mm2,
        ast_mm2=total_ast_mm2,
        axial_cap_factor_value=axial_cap_factor_value,
    )
    df["phi_pn_kN"] = df["phi_pn_kN"].clip(upper=phi_pn_max_kN)
    df = df.drop_duplicates(subset=["phi_mn_kNm", "phi_pn_kN"]).reset_index(drop=True)
    return df


def moment_capacity_at_pu(curve_df: pd.DataFrame, pu_kN: float) -> float:
    if curve_df.empty:
        return 0.0
    rows = curve_df.sort_values(by="phi_pn_kN", ascending=False).reset_index(drop=True)
    points = list(zip(rows["phi_pn_kN"], rows["phi_mn_kNm"]))
    moments: list[float] = []

    for i in range(len(points) - 1):
        p1, m1 = points[i]
        p2, m2 = points[i + 1]
        if (p1 >= pu_kN >= p2) or (p2 >= pu_kN >= p1):
            if abs(p2 - p1) < 1e-9:
                moments.append(max(m1, m2))
            else:
                ratio = (pu_kN - p1) / (p2 - p1)
                moments.append(m1 + ratio * (m2 - m1))

    moments.extend(m for p, m in points if p >= pu_kN)
    if not moments:
        return 0.0
    return max(0.0, max(moments))


def evaluate_load_cases_true(
    load_df: pd.DataFrame,
    curve_x: pd.DataFrame,
    curve_y: pd.DataFrame,
    pmm_responses: list[dict],
    eccentricity_mm: float,
) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    governing: dict | None = None

    for _, row in load_df.iterrows():
        case = str(row["Case"])
        pu = float(row["Pu (kN)"])
        mx_input = float(row["Mx (kN-m)"])
        my_input = float(row["My (kN-m)"])
        mnx = moment_capacity_at_pu(curve_x, pu)
        mny = moment_capacity_at_pu(curve_y, pu)
        mx, my, m_min, min_ecc_applied = enforce_minimum_eccentricity(pu, mx_input, my_input, eccentricity_mm, mnx, mny)
        contour = pmm_slice_from_responses(pmm_responses, pu)
        ratio = ray_polygon_utilization(contour, mx, my)
        status = "OK" if ratio <= 1.0 else "NG"
        item = {
            "Case": case,
            "Pu (kN)": round(pu, 1),
            "Mx input (kN-m)": round(mx_input, 1),
            "My input (kN-m)": round(my_input, 1),
            "Mx used (kN-m)": round(mx, 1),
            "My used (kN-m)": round(my, 1),
            "Mmin from e_min (kN-m)": round(m_min, 1),
            "Min ecc applied": "Yes" if min_ecc_applied else "No",
            "phi Mnx at Pu (kN-m)": round(mnx, 1),
            "phi Mny at Pu (kN-m)": round(mny, 1),
            "PMM ratio": round(ratio, 3) if math.isfinite(ratio) else None,
            "Status": status,
        }
        rows.append(item)
        if governing is None or ratio > governing["ratio"]:
            governing = {
                "case": case,
                "ratio": ratio,
                "mnx": mnx,
                "mny": mny,
                "pu": pu,
                "mx": mx,
                "my": my,
                "mx_input": mx_input,
                "my_input": my_input,
                "m_min": m_min,
                "min_ecc_applied": min_ecc_applied,
                "contour": contour,
            }

    return pd.DataFrame(rows), (
        governing
        or {
            "case": "",
            "ratio": 0.0,
            "mnx": 0.0,
            "mny": 0.0,
            "pu": 0.0,
            "mx": 0.0,
            "my": 0.0,
            "mx_input": 0.0,
            "my_input": 0.0,
            "m_min": 0.0,
            "min_ecc_applied": False,
            "contour": pd.DataFrame(),
        }
    )


@st.cache_data(show_spinner=False)
def solve_required_ast_from_pmm(
    geometry: SectionGeometry,
    shape: Shape,
    cover_mm: float,
    tie_dia_mm: float,
    bar_dia_mm: float,
    fc_mpa: float,
    fy_mpa: float,
    phi: float,
    column_type: str,
    axial_cap_factor_value: float,
    min_ratio: float,
    max_ratio: float,
    load_df: pd.DataFrame,
    bars_width_face: int | None = None,
    bars_depth_face: int | None = None,
    bars_circular: int | None = None,
) -> tuple[float, bool]:
    total_bars = layout_total_bars(shape, bars_width_face, bars_depth_face, bars_circular)
    if total_bars <= 0:
        return float("nan"), False

    ast_min = min_ratio * geometry.ag_mm2
    ast_max = max_ratio * geometry.ag_mm2
    current_ast = total_bars * bar_area_mm2(bar_dia_mm)
    eccentricity_value = minimum_eccentricity_mm(geometry)

    def governing_ratio_for_ast(total_ast_mm2: float) -> float:
        bar_area_single = total_ast_mm2 / total_bars
        curve_x = interaction_curve(
            geometry=geometry,
            shape=shape,
            cover_mm=cover_mm,
            tie_dia_mm=tie_dia_mm,
            bar_dia_mm=bar_dia_mm,
            fc_mpa=fc_mpa,
            fy_mpa=fy_mpa,
            phi=phi,
            column_type=column_type,
            axial_cap_factor_value=axial_cap_factor_value,
            axis="x",
            bar_area_single_override_mm2=bar_area_single,
            bars_width_face=bars_width_face,
            bars_depth_face=bars_depth_face,
            bars_circular=bars_circular,
        )
        curve_y = interaction_curve(
            geometry=geometry,
            shape=shape,
            cover_mm=cover_mm,
            tie_dia_mm=tie_dia_mm,
            bar_dia_mm=bar_dia_mm,
            fc_mpa=fc_mpa,
            fy_mpa=fy_mpa,
            phi=phi,
            column_type=column_type,
            axial_cap_factor_value=axial_cap_factor_value,
            axis="y",
            bar_area_single_override_mm2=bar_area_single,
            bars_width_face=bars_width_face,
            bars_depth_face=bars_depth_face,
            bars_circular=bars_circular,
        )
        pmm_responses = rotated_pmm_responses(
            geometry=geometry,
            shape=shape,
            cover_mm=cover_mm,
            tie_dia_mm=tie_dia_mm,
            bar_dia_mm=bar_dia_mm,
            fc_mpa=fc_mpa,
            fy_mpa=fy_mpa,
            phi=phi,
            column_type=column_type,
            axial_cap_factor_value=axial_cap_factor_value,
            bar_area_single_override_mm2=bar_area_single,
            bars_width_face=bars_width_face,
            bars_depth_face=bars_depth_face,
            bars_circular=bars_circular,
            angle_count=31,
            target_fiber_size_mm=25.0,
        )
        _, governing = evaluate_load_cases_true(load_df, curve_x, curve_y, pmm_responses, eccentricity_value)
        return float(governing["ratio"])

    lower = max(ast_min, 1e-6)
    lower_ratio = governing_ratio_for_ast(lower)
    if lower_ratio <= 1.0:
        return lower, True

    upper = min(max(current_ast, lower), ast_max)
    upper_ratio = governing_ratio_for_ast(upper)
    while upper_ratio > 1.0 and upper < ast_max - 1e-6:
        upper = min(ast_max, max(upper * 1.35, upper + 0.05 * geometry.ag_mm2))
        upper_ratio = governing_ratio_for_ast(upper)

    if upper_ratio > 1.0:
        return float("nan"), False

    for _ in range(12):
        mid = 0.5 * (lower + upper)
        if governing_ratio_for_ast(mid) <= 1.0:
            upper = mid
        else:
            lower = mid
    return upper, True


def analyze_manual_layout(
    geometry: SectionGeometry,
    cover_mm: float,
    tie_dia_mm: float,
    bar_dia_mm: float,
    min_clear_spacing_user_mm: float,
    phi: float,
    fc_mpa: float,
    fy_mpa: float,
    min_ratio: float,
    max_ratio: float,
    shape: Shape,
    axial_cap_factor_value: float,
    ast_required_pmm_mm2: float,
    pmm_required_found: bool,
    governing_pu_kN: float,
    governing_case_name: str,
    governing_ratio: float,
    phi_mnx_at_pu_kNm: float,
    phi_mny_at_pu_kNm: float,
    bars_width_face: int | None = None,
    bars_depth_face: int | None = None,
    bars_circular: int | None = None,
) -> DesignResult:
    min_clear = minimum_clear_spacing_mm(bar_dia_mm, min_clear_spacing_user_mm)
    ast_axial_starter, ast_equation, ast_min = required_steel_area_mm2(
        governing_pu_kN,
        phi,
        fc_mpa,
        fy_mpa,
        geometry.ag_mm2,
        min_ratio,
        axial_cap_factor_value,
    )
    ast_max = max_ratio * geometry.ag_mm2

    if shape == "Circular":
        n_bars = int(bars_circular or 0)
        ast = n_bars * bar_area_mm2(bar_dia_mm)
        spacing_ok, spacing_note = circular_spacing_check(
            diameter_mm=float(geometry.diameter_mm),
            clear_cover_mm=cover_mm,
            tie_dia_mm=tie_dia_mm,
            bar_dia_mm=bar_dia_mm,
            n_bars=n_bars,
            min_clear_spacing_mm=min_clear,
        )
    else:
        n_bars = total_bars_rectangular(int(bars_width_face or 0), int(bars_depth_face or 0))
        ast = n_bars * bar_area_mm2(bar_dia_mm)
        spacing_ok, spacing_note = rectangular_spacing_check(
            width_mm=float(geometry.width_mm),
            depth_mm=float(geometry.depth_mm),
            clear_cover_mm=cover_mm,
            tie_dia_mm=tie_dia_mm,
            bar_dia_mm=bar_dia_mm,
            bars_width_face=int(bars_width_face or 0),
            bars_depth_face=int(bars_depth_face or 0),
            min_clear_spacing_mm=min_clear,
        )
    minimum_ecc_value = minimum_eccentricity_mm(geometry)

    phi_pn = design_strength_kN(phi, fc_mpa, fy_mpa, geometry.ag_mm2, ast, axial_cap_factor_value)
    overall_ok = spacing_ok and ast >= ast_min and ast <= ast_max and governing_ratio <= 1.0
    return DesignResult(
        ast_required_mm2=ast_required_pmm_mm2,
        ast_axial_starter_mm2=ast_axial_starter,
        ast_from_equation_mm2=ast_equation,
        ast_min_mm2=ast_min,
        ast_max_mm2=ast_max,
        phi_pn_kN=phi_pn,
        steel_ratio_percent=100.0 * ast / geometry.ag_mm2,
        spacing_ok=spacing_ok,
        spacing_note=spacing_note,
        overall_ok=overall_ok,
        minimum_eccentricity_mm=minimum_ecc_value,
        governing_ratio=governing_ratio,
        governing_load_case=governing_case_name,
        phi_mnx_at_pu_kNm=phi_mnx_at_pu_kNm,
        phi_mny_at_pu_kNm=phi_mny_at_pu_kNm,
        pmm_required_found=pmm_required_found,
    )


def auto_design_options(
    geometry: SectionGeometry,
    cover_mm: float,
    tie_dia_mm: float,
    fc_mpa: float,
    fy_mpa: float,
    phi: float,
    column_type: str,
    min_ratio: float,
    max_ratio: float,
    min_clear_spacing_user_mm: float,
    shape: Shape,
    bar_sizes_mm: list[float],
    load_df: pd.DataFrame,
    axial_cap_factor_value: float,
) -> pd.DataFrame:
    rows: list[dict] = []
    max_pu_for_prefilter = float(load_df["Pu (kN)"].max()) if not load_df.empty else 0.0
    ast_required_prefilter, _, _ = required_steel_area_mm2(max_pu_for_prefilter, phi, fc_mpa, fy_mpa, geometry.ag_mm2, min_ratio, axial_cap_factor_value)
    eccentricity_value = minimum_eccentricity_mm(geometry)

    for dia in bar_sizes_mm:
        if shape == "Circular":
            for n_bars in range(6, 33):
                ast = n_bars * bar_area_mm2(dia)
                if ast < ast_required_prefilter or ast > max_ratio * geometry.ag_mm2:
                    continue
                curve_x = interaction_curve(geometry, shape, cover_mm, tie_dia_mm, dia, fc_mpa, fy_mpa, phi, column_type, axial_cap_factor_value, "x", bars_circular=n_bars)
                curve_y = interaction_curve(geometry, shape, cover_mm, tie_dia_mm, dia, fc_mpa, fy_mpa, phi, column_type, axial_cap_factor_value, "y", bars_circular=n_bars)
                pmm_responses = rotated_pmm_responses(
                    geometry=geometry,
                    shape=shape,
                    cover_mm=cover_mm,
                    tie_dia_mm=tie_dia_mm,
                    bar_dia_mm=dia,
                    fc_mpa=fc_mpa,
                    fy_mpa=fy_mpa,
                    phi=phi,
                    column_type=column_type,
                    axial_cap_factor_value=axial_cap_factor_value,
                    bars_circular=n_bars,
                    angle_count=19,
                    target_fiber_size_mm=40.0,
                )
                _, governing = evaluate_load_cases_true(load_df, curve_x, curve_y, pmm_responses, eccentricity_value)
                result = analyze_manual_layout(
                    geometry=geometry,
                    cover_mm=cover_mm,
                    tie_dia_mm=tie_dia_mm,
                    bar_dia_mm=dia,
                    min_clear_spacing_user_mm=min_clear_spacing_user_mm,
                    phi=phi,
                    fc_mpa=fc_mpa,
                    fy_mpa=fy_mpa,
                    min_ratio=min_ratio,
                    max_ratio=max_ratio,
                    shape=shape,
                    axial_cap_factor_value=axial_cap_factor_value,
                    ast_required_pmm_mm2=ast,
                    pmm_required_found=True,
                    governing_pu_kN=governing["pu"],
                    governing_case_name=governing["case"],
                    governing_ratio=governing["ratio"],
                    phi_mnx_at_pu_kNm=governing["mnx"],
                    phi_mny_at_pu_kNm=governing["mny"],
                    bars_circular=n_bars,
                )
                if result.spacing_ok and ast <= result.ast_max_mm2 and result.overall_ok:
                    rows.append(
                        {
                            "Bar size (mm)": dia,
                            "Arrangement": f"{n_bars} bars around perimeter",
                            "Total bars": n_bars,
                            "Ast provided (mm2)": round(ast, 1),
                            "Steel ratio (%)": round(result.steel_ratio_percent, 3),
                            "Governing case": governing["case"],
                            "PMM ratio": round(governing["ratio"], 3),
                            "phi Pn (kN)": round(result.phi_pn_kN, 1),
                            "Spacing note": result.spacing_note,
                        }
                    )
        else:
            for bx in range(2, 13):
                for by in range(2, 13):
                    total = total_bars_rectangular(bx, by)
                    if total < 4:
                        continue
                    ast = total * bar_area_mm2(dia)
                    if ast < ast_required_prefilter or ast > max_ratio * geometry.ag_mm2:
                        continue
                    curve_x = interaction_curve(geometry, shape, cover_mm, tie_dia_mm, dia, fc_mpa, fy_mpa, phi, column_type, axial_cap_factor_value, "x", bars_width_face=bx, bars_depth_face=by)
                    curve_y = interaction_curve(geometry, shape, cover_mm, tie_dia_mm, dia, fc_mpa, fy_mpa, phi, column_type, axial_cap_factor_value, "y", bars_width_face=bx, bars_depth_face=by)
                    pmm_responses = rotated_pmm_responses(
                        geometry=geometry,
                        shape=shape,
                        cover_mm=cover_mm,
                        tie_dia_mm=tie_dia_mm,
                        bar_dia_mm=dia,
                        fc_mpa=fc_mpa,
                        fy_mpa=fy_mpa,
                        phi=phi,
                        column_type=column_type,
                        axial_cap_factor_value=axial_cap_factor_value,
                        bars_width_face=bx,
                        bars_depth_face=by,
                        angle_count=19,
                        target_fiber_size_mm=40.0,
                    )
                    _, governing = evaluate_load_cases_true(load_df, curve_x, curve_y, pmm_responses, eccentricity_value)
                    result = analyze_manual_layout(
                        geometry=geometry,
                        cover_mm=cover_mm,
                        tie_dia_mm=tie_dia_mm,
                        bar_dia_mm=dia,
                        min_clear_spacing_user_mm=min_clear_spacing_user_mm,
                        phi=phi,
                        fc_mpa=fc_mpa,
                        fy_mpa=fy_mpa,
                        min_ratio=min_ratio,
                        max_ratio=max_ratio,
                        shape=shape,
                        axial_cap_factor_value=axial_cap_factor_value,
                        ast_required_pmm_mm2=ast,
                        pmm_required_found=True,
                        governing_pu_kN=governing["pu"],
                        governing_case_name=governing["case"],
                        governing_ratio=governing["ratio"],
                        phi_mnx_at_pu_kNm=governing["mnx"],
                        phi_mny_at_pu_kNm=governing["mny"],
                        bars_width_face=bx,
                        bars_depth_face=by,
                    )
                    if result.spacing_ok and ast <= result.ast_max_mm2 and result.overall_ok:
                        rows.append(
                            {
                                "Bar size (mm)": dia,
                                "Arrangement": f"{bx} bars on top/bottom, {by} bars on left/right",
                                "Total bars": total,
                                "Ast provided (mm2)": round(ast, 1),
                                "Steel ratio (%)": round(result.steel_ratio_percent, 3),
                                "Governing case": governing["case"],
                                "PMM ratio": round(governing["ratio"], 3),
                                "phi Pn (kN)": round(result.phi_pn_kN, 1),
                                "Spacing note": result.spacing_note,
                            }
                        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values(by=["Ast provided (mm2)", "Total bars", "Bar size (mm)"]).drop_duplicates().reset_index(drop=True)


def section_figure(
    geometry: SectionGeometry,
    shape: Shape,
    cover_mm: float,
    tie_dia_mm: float,
    bar_dia_mm: float,
    note_text: str = "",
    bars_width_face: int | None = None,
    bars_depth_face: int | None = None,
    bars_circular: int | None = None,
) -> go.Figure:
    fig = go.Figure()
    if shape == "Circular":
        diameter = float(geometry.diameter_mm)
        radius = diameter / 2.0
        theta = [i * 2.0 * math.pi / 180.0 for i in range(361)]
        fig.add_trace(
            go.Scatter(
                x=[radius * math.cos(t) for t in theta],
                y=[radius * math.sin(t) for t in theta],
                mode="lines",
                name="Section",
                line={"color": "#0f172a", "width": 3},
            )
        )
        core_radius = radius - cover_mm - tie_dia_mm
        if core_radius > 0:
            fig.add_trace(
                go.Scatter(
                    x=[core_radius * math.cos(t) for t in theta],
                    y=[core_radius * math.sin(t) for t in theta],
                    mode="lines",
                    name="Tie line",
                    line={"color": "#94a3b8", "dash": "dash"},
                )
            )
        coords = circular_bar_coordinates(diameter, cover_mm, tie_dia_mm, bar_dia_mm, int(bars_circular or 0))
        size_ref = max(10.0, bar_dia_mm * 1.2)
        limit = radius * 1.15
    else:
        width = float(geometry.width_mm)
        depth = float(geometry.depth_mm)
        x_outline = [-width / 2.0, width / 2.0, width / 2.0, -width / 2.0, -width / 2.0]
        y_outline = [-depth / 2.0, -depth / 2.0, depth / 2.0, depth / 2.0, -depth / 2.0]
        fig.add_trace(go.Scatter(x=x_outline, y=y_outline, mode="lines", name="Section", line={"color": "#0f172a", "width": 3}))
        x_tie = [
            -width / 2.0 + cover_mm + tie_dia_mm / 2.0,
            width / 2.0 - cover_mm - tie_dia_mm / 2.0,
            width / 2.0 - cover_mm - tie_dia_mm / 2.0,
            -width / 2.0 + cover_mm + tie_dia_mm / 2.0,
            -width / 2.0 + cover_mm + tie_dia_mm / 2.0,
        ]
        y_tie = [
            -depth / 2.0 + cover_mm + tie_dia_mm / 2.0,
            -depth / 2.0 + cover_mm + tie_dia_mm / 2.0,
            depth / 2.0 - cover_mm - tie_dia_mm / 2.0,
            depth / 2.0 - cover_mm - tie_dia_mm / 2.0,
            -depth / 2.0 + cover_mm + tie_dia_mm / 2.0,
        ]
        fig.add_trace(go.Scatter(x=x_tie, y=y_tie, mode="lines", name="Tie line", line={"color": "#94a3b8", "dash": "dash"}))
        coords = rectangular_bar_coordinates(width, depth, cover_mm, tie_dia_mm, bar_dia_mm, int(bars_width_face or 0), int(bars_depth_face or 0))
        size_ref = max(10.0, bar_dia_mm * 1.2)
        limit = max(width, depth) * 0.65

    if coords:
        fig.add_trace(
            go.Scatter(
                x=[x for x, _ in coords],
                y=[y for _, y in coords],
                mode="markers",
                name="Main bars",
                marker={"size": size_ref, "color": "#dc2626", "line": {"color": "#7f1d1d", "width": 1}},
            )
        )

    fig.update_layout(
        title={"text": "Section Reinforcement", "y": 0.97},
        template="plotly_white",
        height=520,
        xaxis={"title": "x (mm)", "scaleanchor": "y", "range": [-limit, limit]},
        yaxis={"title": "y (mm)", "range": [-limit, limit]},
        legend={"orientation": "h"},
        margin={"l": 20, "r": 20, "t": 80, "b": 20},
    )
    if note_text:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.98,
            text=note_text,
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.92)",
            bordercolor="#cbd5e1",
            borderwidth=1,
        )
    return fig


def axial_capacity_plot(result: DesignResult, pu_kN: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Bar(x=["Demand Pu", "Capacity phi Pn"], y=[pu_kN, result.phi_pn_kN], marker_color=["#f97316", "#16a34a"]))
    fig.update_layout(template="plotly_white", height=360, title="Axial Demand vs Capacity", yaxis_title="kN", margin={"l": 20, "r": 20, "t": 50, "b": 20})
    return fig


def interaction_plot(df_x: pd.DataFrame, df_y: pd.DataFrame, pu_kN: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_x["phi_mn_kNm"], y=df_x["phi_pn_kN"], mode="lines", name="P-Mx"))
    fig.add_trace(go.Scatter(x=df_y["phi_mn_kNm"], y=df_y["phi_pn_kN"], mode="lines", name="P-My"))
    fig.add_hline(y=pu_kN, line_dash="dash", line_color="#dc2626", annotation_text=f"Pu = {pu_kN:,.0f} kN")
    fig.update_layout(template="plotly_white", height=460, title="Strain Compatibility Interaction Curves", xaxis_title="phi Mn (kN-m)", yaxis_title="phi Pn (kN)")
    return fig


def section_note_text(
    shape: Shape,
    bar_dia_mm: float,
    total_bars: int,
    ast_mm2: float,
    rho_percent: float,
    spacing_note: str,
    bars_width_face: int | None = None,
    bars_depth_face: int | None = None,
    bars_circular: int | None = None,
) -> str:
    if shape == "Circular":
        arrangement = f"{int(bars_circular or 0)} bars around perimeter"
    else:
        arrangement = f"Top/bottom: {int(bars_width_face or 0)} each, left/right: {int(bars_depth_face or 0)} each"
    return (
        f"{arrangement}<br>"
        f"Main bars: DB{int(bar_dia_mm)} | total bars = {total_bars}<br>"
        f"Ast = {ast_mm2:,.0f} mm2 | rho = {rho_percent:.3f}%<br>"
        f"{spacing_note}"
    )


def interaction_plot_with_demand(df_x: pd.DataFrame, df_y: pd.DataFrame, pu_kN: float, mx_kNm: float, my_kNm: float) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_x["phi_mn_kNm"], y=df_x["phi_pn_kN"], mode="lines", name="about x", line={"color": "#2563eb", "width": 3}))
    fig.add_trace(go.Scatter(x=df_y["phi_mn_kNm"], y=df_y["phi_pn_kN"], mode="lines", name="about y", line={"color": "#e11d48", "width": 3}))
    fig.add_trace(
        go.Scatter(
            x=[abs(mx_kNm)],
            y=[pu_kN],
            mode="markers",
            name="Pu, Mux",
            marker={"symbol": "x", "size": 10, "color": "#2563eb", "line": {"width": 2}},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[abs(my_kNm)],
            y=[pu_kN],
            mode="markers",
            name="Pu, Muy",
            marker={"symbol": "x", "size": 10, "color": "#e11d48", "line": {"width": 2}},
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=460,
        title={"text": "Uniaxial Interaction Curves", "y": 0.97},
        xaxis_title="phi Mn (kN-m)",
        yaxis_title="phi Pn (kN)",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.06, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 90, "b": 20},
    )
    return fig


def concrete_fibers(
    geometry: SectionGeometry,
    shape: Shape,
    target_fiber_size_mm: float = 25.0,
) -> list[tuple[float, float, float]]:
    fibers: list[tuple[float, float, float]] = []
    if shape == "Circular":
        diameter = float(geometry.diameter_mm)
        radius = diameter / 2.0
        divisions = max(24, min(60, int(math.ceil(diameter / target_fiber_size_mm))))
        step = diameter / divisions
        start = -radius + 0.5 * step
        for i in range(divisions):
            x = start + i * step
            for j in range(divisions):
                y = start + j * step
                if x * x + y * y <= radius * radius:
                    fibers.append((x, y, step * step))
        return fibers

    width = float(geometry.width_mm)
    depth = float(geometry.depth_mm)
    nx = max(20, min(60, int(math.ceil(width / target_fiber_size_mm))))
    ny = max(20, min(60, int(math.ceil(depth / target_fiber_size_mm))))
    dx = width / nx
    dy = depth / ny
    x0 = -width / 2.0 + 0.5 * dx
    y0 = -depth / 2.0 + 0.5 * dy
    for i in range(nx):
        x = x0 + i * dx
        for j in range(ny):
            y = y0 + j * dy
            fibers.append((x, y, dx * dy))
    return fibers


def section_support_coordinate(geometry: SectionGeometry, shape: Shape, theta_rad: float) -> float:
    if shape == "Circular":
        return float(geometry.diameter_mm) / 2.0
    width = float(geometry.width_mm)
    depth = float(geometry.depth_mm)
    return 0.5 * width * abs(math.cos(theta_rad)) + 0.5 * depth * abs(math.sin(theta_rad))


def rotated_section_response(
    geometry: SectionGeometry,
    shape: Shape,
    theta_rad: float,
    neutral_axis_mm: float,
    fc_mpa: float,
    fy_mpa: float,
    phi: float,
    column_type: str,
    axial_cap_factor_value: float,
    fibers: list[tuple[float, float, float]],
    bar_coords: list[tuple[float, float]],
    bar_area_single_mm2: float,
) -> dict[str, float]:
    eps_cu = 0.003
    beta1 = beta1_aci(fc_mpa)
    support = section_support_coordinate(geometry, shape, theta_rad)
    a_depth = min(beta1 * neutral_axis_mm, 2.0 * support)
    threshold = support - a_depth
    cos_t = math.cos(theta_rad)
    sin_t = math.sin(theta_rad)

    concrete_force_n = 0.0
    mx_nmm = 0.0
    my_nmm = 0.0
    for x, y, area in fibers:
        u = x * cos_t + y * sin_t
        if u >= threshold:
            force = 0.85 * fc_mpa * area
            concrete_force_n += force
            mx_nmm += force * y
            my_nmm += force * x

    steel_force_n = 0.0
    max_tension_strain = 0.0
    for x, y in bar_coords:
        u = x * cos_t + y * sin_t
        dist_from_face = support - u
        strain = eps_cu * (1.0 - dist_from_face / neutral_axis_mm)
        max_tension_strain = max(max_tension_strain, -strain)
        stress = steel_stress_mpa(strain, fy_mpa)
        in_compression_block = u >= threshold
        force = bar_area_single_mm2 * (stress - 0.85 * fc_mpa) if in_compression_block else bar_area_single_mm2 * stress
        steel_force_n += force
        mx_nmm += force * y
        my_nmm += force * x

    nominal_p_n = concrete_force_n + steel_force_n
    phi_value = strain_based_phi(column_type, fy_mpa, max_tension_strain)
    phi_pn_kN = phi_value * nominal_p_n / 1000.0
    phi_pn_max_kN = design_strength_kN(
        phi=phi,
        fc_mpa=fc_mpa,
        fy_mpa=fy_mpa,
        ag_mm2=geometry.ag_mm2,
        ast_mm2=len(bar_coords) * bar_area_single_mm2,
        axial_cap_factor_value=axial_cap_factor_value,
    )
    return {
        "phi_pn_kN": min(phi_pn_kN, phi_pn_max_kN),
        "phi_mx_kNm": phi_value * mx_nmm / 1_000_000.0,
        "phi_my_kNm": phi_value * my_nmm / 1_000_000.0,
        "phi_value": phi_value,
        "max_tension_strain": max_tension_strain,
    }


def pmm_sample_fractions() -> list[float]:
    return [0.02, 0.05, 0.08, 0.12, 0.16, 0.22, 0.30, 0.40, 0.55, 0.75, 1.0, 1.35, 1.8, 2.4, 3.2, 4.5, 6.0]


def rotated_pmm_responses(
    geometry: SectionGeometry,
    shape: Shape,
    cover_mm: float,
    tie_dia_mm: float,
    bar_dia_mm: float,
    fc_mpa: float,
    fy_mpa: float,
    phi: float,
    column_type: str,
    axial_cap_factor_value: float,
    bar_area_single_override_mm2: float | None = None,
    bars_width_face: int | None = None,
    bars_depth_face: int | None = None,
    bars_circular: int | None = None,
    angle_count: int = 49,
    target_fiber_size_mm: float = 25.0,
) -> list[dict]:
    fibers = concrete_fibers(geometry, shape, target_fiber_size_mm=target_fiber_size_mm)
    bar_coords = section_bar_coordinates(
        geometry=geometry,
        shape=shape,
        cover_mm=cover_mm,
        tie_dia_mm=tie_dia_mm,
        bar_dia_mm=bar_dia_mm,
        bars_width_face=bars_width_face,
        bars_depth_face=bars_depth_face,
        bars_circular=bars_circular,
    )
    bar_area_single = bar_area_single_override_mm2 if bar_area_single_override_mm2 is not None else bar_area_mm2(bar_dia_mm)
    responses: list[dict] = []
    for angle_index in range(angle_count):
        theta = 2.0 * math.pi * angle_index / angle_count
        support = section_support_coordinate(geometry, shape, theta)
        for fraction in pmm_sample_fractions():
            c_value = max(1.0, 2.0 * support * fraction)
            response = rotated_section_response(
                geometry=geometry,
                shape=shape,
                theta_rad=theta,
                neutral_axis_mm=c_value,
                fc_mpa=fc_mpa,
                fy_mpa=fy_mpa,
                phi=phi,
                column_type=column_type,
                axial_cap_factor_value=axial_cap_factor_value,
                fibers=fibers,
                bar_coords=bar_coords,
                bar_area_single_mm2=bar_area_single,
            )
            responses.append(
                {
                    "theta_rad": theta,
                    "c_mm": c_value,
                    "phi_pn_kN": response["phi_pn_kN"],
                    "phi_mx_kNm": response["phi_mx_kNm"],
                    "phi_my_kNm": response["phi_my_kNm"],
                    "phi_value": response["phi_value"],
                    "max_tension_strain": response["max_tension_strain"],
                }
            )
    return responses


def pmm_slice_from_responses(responses: list[dict], pu_kN: float) -> pd.DataFrame:
    if not responses:
        return pd.DataFrame(columns=["mx_kNm", "my_kNm", "theta_rad"])
    rows: list[dict[str, float]] = []
    theta_values = sorted({item["theta_rad"] for item in responses})
    for theta in theta_values:
        series = [item for item in responses if abs(item["theta_rad"] - theta) < 1e-9]
        series = sorted(series, key=lambda item: item["phi_pn_kN"], reverse=True)
        found = False
        for first, second in zip(series, series[1:]):
            p1 = first["phi_pn_kN"]
            p2 = second["phi_pn_kN"]
            if (p1 >= pu_kN >= p2) or (p2 >= pu_kN >= p1):
                ratio = 0.0 if abs(p2 - p1) < 1e-9 else (pu_kN - p1) / (p2 - p1)
                rows.append(
                    {
                        "theta_rad": theta,
                        "mx_kNm": first["phi_mx_kNm"] + ratio * (second["phi_mx_kNm"] - first["phi_mx_kNm"]),
                        "my_kNm": first["phi_my_kNm"] + ratio * (second["phi_my_kNm"] - first["phi_my_kNm"]),
                    }
                )
                found = True
                break
        if not found:
            above = [item for item in series if item["phi_pn_kN"] >= pu_kN]
            if above:
                best = max(above, key=lambda item: math.hypot(item["phi_mx_kNm"], item["phi_my_kNm"]))
                rows.append({"theta_rad": theta, "mx_kNm": best["phi_mx_kNm"], "my_kNm": best["phi_my_kNm"]})
    if not rows:
        return pd.DataFrame(columns=["mx_kNm", "my_kNm", "theta_rad"])
    rows.append(rows[0].copy())
    return pd.DataFrame(rows)


def ray_polygon_utilization(contour_df: pd.DataFrame, demand_mx_kNm: float, demand_my_kNm: float) -> float:
    if contour_df.empty:
        return float("inf")
    if abs(demand_mx_kNm) < 1e-9 and abs(demand_my_kNm) < 1e-9:
        return 0.0
    direction = (demand_mx_kNm, demand_my_kNm)
    best_t: float | None = None
    points = list(zip(contour_df["mx_kNm"], contour_df["my_kNm"]))
    for (ax, ay), (bx, by) in zip(points, points[1:]):
        sx = bx - ax
        sy = by - ay
        denominator = direction[0] * sy - direction[1] * sx
        if abs(denominator) < 1e-9:
            continue
        t = (ax * sy - ay * sx) / denominator
        u = (ax * direction[1] - ay * direction[0]) / denominator
        if t >= 0.0 and 0.0 <= u <= 1.0:
            if best_t is None or t < best_t:
                best_t = t
    if best_t is None or best_t <= 0.0:
        return float("inf")
    return 1.0 / best_t


def pmm_slice_plot(contour_df: pd.DataFrame, pu_kN: float, mx_kNm: float, my_kNm: float, ratio: float) -> go.Figure:
    contour = contour_df.copy()
    fig = go.Figure()
    if not contour.empty:
        fig.add_trace(
            go.Scatter(
                x=contour["mx_kNm"],
                y=contour["my_kNm"],
                mode="lines",
                name="PMM slice",
                line={"color": "#0284c7", "width": 2.5},
                fill="toself",
                fillcolor="rgba(2, 132, 199, 0.10)",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0.0, mx_kNm],
            y=[0.0, my_kNm],
            mode="lines+markers",
            name="demand",
            line={"color": "#0f766e", "width": 2},
            marker={"size": 6, "color": "#0f766e"},
        )
    )
    fig.add_annotation(
        x=mx_kNm,
        y=my_kNm,
        text=f"PMM U = {ratio:.3f}",
        showarrow=True,
        arrowhead=2,
        ax=30,
        ay=-20,
    )
    max_x = max(abs(mx_kNm), abs(contour["mx_kNm"]).max() if not contour.empty else 0.0, 1.0)
    max_y = max(abs(my_kNm), abs(contour["my_kNm"]).max() if not contour.empty else 0.0, 1.0)
    fig.update_layout(
        template="plotly_white",
        height=420,
        title=f"PMM Mux-My slice at Pu = {pu_kN:,.0f} kN",
        xaxis={"title": "Mux (kN-m)", "zeroline": True, "range": [-1.15 * max_x, 1.15 * max_x], "scaleanchor": "y"},
        yaxis={"title": "Muy (kN-m)", "zeroline": True, "range": [-1.15 * max_y, 1.15 * max_y]},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
    )
    return fig


def pmm_surface_plot(
    responses: list[dict],
    contour_df: pd.DataFrame,
    pu_kN: float,
    mx_kNm: float,
    my_kNm: float,
    ratio: float,
) -> go.Figure:
    if not responses:
        return go.Figure()
    theta_count = len(sorted({item["theta_rad"] for item in responses}))
    expected_points = theta_count + 1
    p_max = max(item["phi_pn_kN"] for item in responses)
    p_levels = [p_max * i / 24.0 for i in range(1, 25)]
    x_grid: list[list[float]] = []
    y_grid: list[list[float]] = []
    z_grid: list[list[float]] = []
    for p_level in p_levels:
        contour_at_p = pmm_slice_from_responses(responses, p_level)
        if contour_at_p.empty or len(contour_at_p) != expected_points:
            continue
        x_grid.append(contour_at_p["mx_kNm"].tolist())
        y_grid.append(contour_at_p["my_kNm"].tolist())
        z_grid.append([p_level] * len(contour_at_p))

    current_slice = contour_df.copy()
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=x_grid,
            y=y_grid,
            z=z_grid,
            showscale=False,
            opacity=0.62,
            colorscale=[[0.0, "#bfdbfe"], [1.0, "#60a5fa"]],
            name="PMM surface",
            hovertemplate="Mux=%{x:.1f}<br>Muy=%{y:.1f}<br>Pu=%{z:.1f}<extra></extra>",
        )
    )
    if not current_slice.empty:
        fig.add_trace(
            go.Scatter3d(
                x=current_slice["mx_kNm"],
                y=current_slice["my_kNm"],
                z=[pu_kN] * len(current_slice),
                mode="lines",
                name="current Pu slice",
                line={"color": "#0284c7", "width": 5},
            )
        )
    fig.add_trace(
        go.Scatter3d(
            x=[0.0, mx_kNm],
            y=[0.0, my_kNm],
            z=[0.0, pu_kN],
            mode="lines",
            name="demand vector",
            line={"color": "#0f766e", "width": 6},
        )
    )
    fig.add_trace(
        go.Scatter3d(
            x=[mx_kNm],
            y=[my_kNm],
            z=[pu_kN],
            mode="markers+text",
            name="load point",
            marker={"size": 5, "color": "#115e59"},
            text=[f"U={ratio:.3f}"],
            textposition="top center",
        )
    )
    fig.update_layout(
        template="plotly_white",
        height=520,
        title="3D PMM interaction surface with load point",
        scene={
            "xaxis_title": "Mux (kN-m)",
            "yaxis_title": "Muy (kN-m)",
            "zaxis_title": "Pu (kN)",
            "camera": {"eye": {"x": 1.6, "y": 1.4, "z": 1.2}},
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0.0},
        margin={"l": 10, "r": 10, "t": 60, "b": 10},
    )
    return fig


def _nonclosing_points(contour_df: pd.DataFrame) -> pd.DataFrame:
    if contour_df.empty:
        return contour_df
    if len(contour_df) > 1 and abs(contour_df.iloc[0]["mx_kNm"] - contour_df.iloc[-1]["mx_kNm"]) < 1e-9 and abs(contour_df.iloc[0]["my_kNm"] - contour_df.iloc[-1]["my_kNm"]) < 1e-9:
        return contour_df.iloc[:-1].copy()
    return contour_df.copy()


@st.cache_data(show_spinner=False)
def run_verification_suite() -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    column_type = "Tied"
    compression_phi = phi_factor(column_type)
    cap = axial_cap_factor(column_type)

    square_geo = section_geometry("Square", 400.0, 400.0, None)
    square_curve_x = interaction_curve(square_geo, "Square", 50.0, 10.0, 20.0, 28.0, 420.0, compression_phi, column_type, cap, "x", bars_width_face=4, bars_depth_face=4)
    square_curve_y = interaction_curve(square_geo, "Square", 50.0, 10.0, 20.0, 28.0, 420.0, compression_phi, column_type, cap, "y", bars_width_face=4, bars_depth_face=4)
    square_pu = 0.4 * min(float(square_curve_x["phi_pn_kN"].max()), float(square_curve_y["phi_pn_kN"].max()))
    square_mnx = moment_capacity_at_pu(square_curve_x, square_pu)
    square_mny = moment_capacity_at_pu(square_curve_y, square_pu)
    square_diff = abs(square_mnx - square_mny) / max(square_mnx, square_mny, 1e-9)
    rows.append(
        {
            "Benchmark": "Square symmetry",
            "Check": "Mnx(Pu) vs Mny(Pu)",
            "Result": f"{square_mnx:,.1f} vs {square_mny:,.1f} kN-m",
            "Error": f"{square_diff * 100.0:.2f}%",
            "Status": "OK" if square_diff <= 0.05 else "Review",
        }
    )

    rect_geo = section_geometry("Rectangle", 300.0, 500.0, None)
    rect_curve_x = interaction_curve(rect_geo, "Rectangle", 50.0, 10.0, 20.0, 28.0, 420.0, compression_phi, column_type, cap, "x", bars_width_face=4, bars_depth_face=4)
    rect_curve_y = interaction_curve(rect_geo, "Rectangle", 50.0, 10.0, 20.0, 28.0, 420.0, compression_phi, column_type, cap, "y", bars_width_face=4, bars_depth_face=4)
    rect_responses = rotated_pmm_responses(rect_geo, "Rectangle", 50.0, 10.0, 20.0, 28.0, 420.0, compression_phi, column_type, cap, bars_width_face=4, bars_depth_face=4, angle_count=72, target_fiber_size_mm=25.0)
    rect_pu = 0.4 * min(float(rect_curve_x["phi_pn_kN"].max()), float(rect_curve_y["phi_pn_kN"].max()))
    rect_slice = _nonclosing_points(pmm_slice_from_responses(rect_responses, rect_pu))
    rect_mnx = moment_capacity_at_pu(rect_curve_x, rect_pu)
    rect_mny = moment_capacity_at_pu(rect_curve_y, rect_pu)
    rect_slice_mx = abs(rect_slice["mx_kNm"]).max() if not rect_slice.empty else 0.0
    rect_slice_my = abs(rect_slice["my_kNm"]).max() if not rect_slice.empty else 0.0
    rect_error = max(abs(rect_slice_mx - rect_mnx) / max(rect_mnx, 1e-9), abs(rect_slice_my - rect_mny) / max(rect_mny, 1e-9))
    rows.append(
        {
            "Benchmark": "Rectangle axis consistency",
            "Check": "PMM slice intercepts vs uniaxial curves",
            "Result": f"x: {rect_slice_mx:,.1f}/{rect_mnx:,.1f}, y: {rect_slice_my:,.1f}/{rect_mny:,.1f}",
            "Error": f"{rect_error * 100.0:.2f}%",
            "Status": "OK" if rect_error <= 0.10 else "Review",
        }
    )

    circ_geo = section_geometry("Circular", None, None, 600.0)
    circ_responses = rotated_pmm_responses(circ_geo, "Circular", 50.0, 10.0, 20.0, 28.0, 420.0, compression_phi, column_type, cap, bars_circular=8, angle_count=72, target_fiber_size_mm=20.0)
    circ_curve_x = interaction_curve(circ_geo, "Circular", 50.0, 10.0, 20.0, 28.0, 420.0, compression_phi, column_type, cap, "x", bars_circular=8)
    circ_curve_y = interaction_curve(circ_geo, "Circular", 50.0, 10.0, 20.0, 28.0, 420.0, compression_phi, column_type, cap, "y", bars_circular=8)
    circ_pu = 0.4 * min(float(circ_curve_x["phi_pn_kN"].max()), float(circ_curve_y["phi_pn_kN"].max()))
    circ_slice = _nonclosing_points(pmm_slice_from_responses(circ_responses, circ_pu))
    radii = [math.hypot(float(mx), float(my)) for mx, my in zip(circ_slice["mx_kNm"], circ_slice["my_kNm"])] if not circ_slice.empty else []
    circ_cov = (pd.Series(radii).std() / max(pd.Series(radii).mean(), 1e-9)) if radii else float("inf")
    rows.append(
        {
            "Benchmark": "Circular rotational invariance",
            "Check": "Radius variation on PMM slice",
            "Result": f"mean R = {pd.Series(radii).mean():,.1f} kN-m" if radii else "No slice",
            "Error": f"{circ_cov * 100.0:.2f}%",
            "Status": "OK" if circ_cov <= 0.08 else "Review",
        }
    )

    return pd.DataFrame(rows)


def clean_load_cases(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ["Case", "Pu (kN)", "Mx (kN-m)", "My (kN-m)"]
    working = df.copy()
    for col in required_cols:
        if col not in working.columns:
            working[col] = 0.0 if col != "Case" else ""
    working["Case"] = working["Case"].astype(str).replace("nan", "")
    for col in required_cols[1:]:
        working[col] = pd.to_numeric(working[col], errors="coerce").fillna(0.0)
    working = working[required_cols]
    working = working[working["Case"].str.strip() != ""].reset_index(drop=True)
    if working.empty:
        working = default_load_cases()
    return working


def parse_available_bar_sizes(text: str) -> list[float]:
    parsed: list[float] = []
    for raw_item in text.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            value = float(item)
        except ValueError:
            st.warning(f"'{item}' is not a valid bar diameter and was skipped.")
            continue
        if value <= 0.0:
            st.warning(f"'{item}' must be greater than 0 and was skipped.")
            continue
        parsed.append(value)
    return sorted(set(parsed))


def json_ready(value):
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


def project_payload(data: dict) -> str:
    return json.dumps({key: json_ready(value) for key, value in data.items()}, indent=2)


st.set_page_config(page_title="Pile Reinforcement Designer", page_icon="P", layout="wide", initial_sidebar_state="expanded")

st.title("Pile Reinforcement Designer")
st.caption("Preliminary RC pile longitudinal reinforcement design for square, rectangular, and circular sections using strain compatibility interaction curves, true PMM checks, auto design, manual check, layout plots, and save/load workflow.")

with st.sidebar:
    st.markdown("### Save / Load Design")
    default_state = {
        "shape": "Square",
        "width_mm": 300.0,
        "depth_mm": 400.0,
        "diameter_mm": 400.0,
        "fc_mpa": 28.0,
        "fy_mpa": 420.0,
        "cover_mm": 50.0,
        "tie_dia_mm": 9.0,
        "column_type": "Tied",
        "min_ratio_percent": 1.0,
        "max_ratio_percent": 8.0,
        "min_clear_spacing_mm": 40.0,
        "mode": "Auto design",
        "manual_bar_dia_mm": 20.0,
        "bars_width_face": 4,
        "bars_depth_face": 4,
        "bars_circular": 8,
        "available_sizes_text": "12, 16, 20, 25, 28, 32",
        "load_cases": default_load_cases().to_dict(orient="records"),
    }
    payload = project_payload({key: st.session_state.get(key, value) for key, value in default_state.items()})
    st.download_button("Save Project", payload, file_name="pile_rebar_design.json", mime="application/json", use_container_width=True)
    uploaded = st.file_uploader("Open project JSON", type=["json"])
    if uploaded is not None:
        try:
            loaded = json.loads(uploaded.getvalue().decode("utf-8"))
            for key, value in loaded.items():
                st.session_state[key] = value
            st.success("Project file loaded.")
        except Exception as exc:
            st.error(f"Could not open project file: {exc}")

    st.header("Design Basis")
    code_basis = st.selectbox("Reference basis", ["Strain Compatibility PMM (ACI 318-19 style)"], index=0)
    column_type = st.radio("Transverse reinforcement type", ["Tied", "Spiral"], horizontal=True, key="column_type")
    min_ratio_percent = st.number_input("Minimum rho for auto (%)", min_value=0.1, max_value=8.0, step=0.1, key="min_ratio_percent")
    max_ratio_percent = st.number_input("Maximum rho for auto (%)", min_value=1.0, max_value=12.0, step=0.1, key="max_ratio_percent")
    min_clear_spacing_user_mm = st.number_input("Minimum clear spacing (mm)", min_value=25.0, max_value=200.0, step=5.0, key="min_clear_spacing_mm")
    tie_dia_mm = st.number_input("Tie / spiral diameter (mm)", min_value=6.0, max_value=20.0, step=1.0, key="tie_dia_mm")
    available_sizes_text = st.text_input("Available main bar diameters (mm)", key="available_sizes_text")

    st.header("Geometry")
    shape = st.selectbox("Pile cross-section", ["Square", "Rectangle", "Circular"], key="shape")
    if shape == "Square":
        width_mm = st.number_input("Width = Depth (mm)", min_value=150.0, step=25.0, key="width_mm")
        depth_mm = width_mm
        diameter_mm = None
    elif shape == "Rectangle":
        width_mm = st.number_input("Width b (mm)", min_value=150.0, step=25.0, key="width_mm")
        depth_mm = st.number_input("Depth h (mm)", min_value=150.0, step=25.0, key="depth_mm")
        diameter_mm = None
    else:
        diameter_mm = st.number_input("Diameter D (mm)", min_value=150.0, step=25.0, key="diameter_mm")
        width_mm = None
        depth_mm = None
    cover_mm = st.number_input("Clear cover to tie / spiral (mm)", min_value=25.0, step=5.0, key="cover_mm")

    st.header("Material and Load")
    fc_mpa = st.number_input("Concrete strength f'c (MPa)", min_value=15.0, max_value=80.0, step=1.0, key="fc_mpa")
    fy_mpa = st.number_input("Steel yield strength fy (MPa)", min_value=240.0, max_value=600.0, step=10.0, key="fy_mpa")

    st.header("Reinforcement")
    mode = st.radio("Reinforcement mode", ["Auto design", "Manual check"], horizontal=True, key="mode")
    manual_bar_dia_mm = st.selectbox("Main bar diameter", DEFAULT_BAR_SIZES_MM, index=2, key="manual_bar_dia_mm")
    if shape == "Circular":
        bars_circular = st.number_input("Bars around perimeter", min_value=6, max_value=40, step=1, key="bars_circular")
        bars_width_face = None
        bars_depth_face = None
    else:
        bars_width_face = st.number_input("Bars on top/bottom faces", min_value=2, max_value=20, step=1, key="bars_width_face")
        bars_depth_face = st.number_input("Bars on left/right faces", min_value=2, max_value=20, step=1, key="bars_depth_face")
        bars_circular = None

st.subheader("Internal Force Load Cases")
if "load_cases" not in st.session_state:
    st.session_state.load_cases = default_load_cases()
edited_load_cases = st.data_editor(
    clean_load_cases(pd.DataFrame(st.session_state.load_cases)),
    num_rows="dynamic",
    use_container_width=True,
    hide_index=True,
    column_config={
        "Case": st.column_config.TextColumn("Case"),
        "Pu (kN)": st.column_config.NumberColumn("Pu (kN)", step=50.0, format="%.1f"),
        "Mx (kN-m)": st.column_config.NumberColumn("Mx (kN-m)", step=10.0, format="%.1f"),
        "My (kN-m)": st.column_config.NumberColumn("My (kN-m)", step=10.0, format="%.1f"),
    },
    key="load_cases_editor",
)
load_cases_df = clean_load_cases(pd.DataFrame(edited_load_cases))
st.session_state.load_cases = load_cases_df.copy()

geometry = section_geometry(shape, width_mm, depth_mm, diameter_mm)
phi = phi_factor(column_type)
axial_cap = axial_cap_factor(column_type)
min_ratio = min_ratio_percent / 100.0
max_ratio = max_ratio_percent / 100.0
eccentricity_value = minimum_eccentricity_mm(geometry)
bars_width_face_int = int(bars_width_face) if bars_width_face is not None else None
bars_depth_face_int = int(bars_depth_face) if bars_depth_face is not None else None
bars_circular_int = int(bars_circular) if bars_circular is not None else None

available_sizes = parse_available_bar_sizes(available_sizes_text)
if not available_sizes:
    available_sizes = DEFAULT_BAR_SIZES_MM.copy()

curve_x_manual = interaction_curve(
    geometry=geometry,
    shape=shape,
    cover_mm=cover_mm,
    tie_dia_mm=tie_dia_mm,
    bar_dia_mm=float(manual_bar_dia_mm),
    fc_mpa=fc_mpa,
    fy_mpa=fy_mpa,
    phi=phi,
    column_type=column_type,
    axial_cap_factor_value=axial_cap,
    axis="x",
    bars_width_face=bars_width_face_int,
    bars_depth_face=bars_depth_face_int,
    bars_circular=bars_circular_int,
)
curve_y_manual = interaction_curve(
    geometry=geometry,
    shape=shape,
    cover_mm=cover_mm,
    tie_dia_mm=tie_dia_mm,
    bar_dia_mm=float(manual_bar_dia_mm),
    fc_mpa=fc_mpa,
    fy_mpa=fy_mpa,
    phi=phi,
    column_type=column_type,
    axial_cap_factor_value=axial_cap,
    axis="y",
    bars_width_face=bars_width_face_int,
    bars_depth_face=bars_depth_face_int,
    bars_circular=bars_circular_int,
)
pmm_responses_true = rotated_pmm_responses(
    geometry=geometry,
    shape=shape,
    cover_mm=cover_mm,
    tie_dia_mm=tie_dia_mm,
    bar_dia_mm=float(manual_bar_dia_mm),
    fc_mpa=fc_mpa,
    fy_mpa=fy_mpa,
    phi=phi,
    column_type=column_type,
    axial_cap_factor_value=axial_cap,
    bars_width_face=bars_width_face_int,
    bars_depth_face=bars_depth_face_int,
    bars_circular=bars_circular_int,
)
load_case_results_df, governing_case = evaluate_load_cases_true(load_cases_df, curve_x_manual, curve_y_manual, pmm_responses_true, eccentricity_value)
pmm_required_ast_mm2, pmm_required_found = solve_required_ast_from_pmm(
    geometry=geometry,
    shape=shape,
    cover_mm=cover_mm,
    tie_dia_mm=tie_dia_mm,
    bar_dia_mm=float(manual_bar_dia_mm),
    fc_mpa=fc_mpa,
    fy_mpa=fy_mpa,
    phi=phi,
    column_type=column_type,
    axial_cap_factor_value=axial_cap,
    min_ratio=min_ratio,
    max_ratio=max_ratio,
    load_df=load_cases_df,
    bars_width_face=bars_width_face_int,
    bars_depth_face=bars_depth_face_int,
    bars_circular=bars_circular_int,
)

manual_result = analyze_manual_layout(
    geometry=geometry,
    cover_mm=cover_mm,
    tie_dia_mm=tie_dia_mm,
    bar_dia_mm=float(manual_bar_dia_mm),
    min_clear_spacing_user_mm=min_clear_spacing_user_mm,
    phi=phi,
    fc_mpa=fc_mpa,
    fy_mpa=fy_mpa,
    min_ratio=min_ratio,
    max_ratio=max_ratio,
    shape=shape,
    axial_cap_factor_value=axial_cap,
    ast_required_pmm_mm2=pmm_required_ast_mm2,
    pmm_required_found=pmm_required_found,
    governing_pu_kN=governing_case["pu"],
    governing_case_name=governing_case["case"],
    governing_ratio=governing_case["ratio"],
    phi_mnx_at_pu_kNm=governing_case["mnx"],
    phi_mny_at_pu_kNm=governing_case["mny"],
    bars_width_face=bars_width_face_int,
    bars_depth_face=bars_depth_face_int,
    bars_circular=bars_circular_int,
)

total_bars = layout_total_bars(shape, bars_width_face_int, bars_depth_face_int, bars_circular_int)
ast_manual = total_bars * bar_area_mm2(float(manual_bar_dia_mm))
required_ast_display = format_required_ast(manual_result.ast_required_mm2, manual_result.pmm_required_found, manual_result.ast_max_mm2)
section_note = section_note_text(
    shape=shape,
    bar_dia_mm=float(manual_bar_dia_mm),
    total_bars=total_bars,
    ast_mm2=ast_manual,
    rho_percent=manual_result.steel_ratio_percent,
    spacing_note=manual_result.spacing_note,
    bars_width_face=bars_width_face_int,
    bars_depth_face=bars_depth_face_int,
    bars_circular=bars_circular_int,
)

section_layout_fig = section_figure(
    geometry=geometry,
    shape=shape,
    cover_mm=cover_mm,
    tie_dia_mm=tie_dia_mm,
    bar_dia_mm=float(manual_bar_dia_mm),
    note_text=section_note,
    bars_width_face=bars_width_face_int,
    bars_depth_face=bars_depth_face_int,
    bars_circular=bars_circular_int,
)
uniaxial_fig = interaction_plot_with_demand(curve_x_manual, curve_y_manual, governing_case["pu"], governing_case["mx"], governing_case["my"])
pmm_slice_true_df = governing_case["contour"].copy() if isinstance(governing_case.get("contour"), pd.DataFrame) else pmm_slice_from_responses(pmm_responses_true, governing_case["pu"])
pmm_true_ratio = governing_case["ratio"]
pmm_slice_fig = pmm_slice_plot(
    contour_df=pmm_slice_true_df,
    pu_kN=governing_case["pu"],
    mx_kNm=governing_case["mx"],
    my_kNm=governing_case["my"],
    ratio=pmm_true_ratio,
)
pmm_surface_fig = pmm_surface_plot(
    responses=pmm_responses_true,
    contour_df=pmm_slice_true_df,
    pu_kN=governing_case["pu"],
    mx_kNm=governing_case["mx"],
    my_kNm=governing_case["my"],
    ratio=pmm_true_ratio,
)
verification_df = run_verification_suite()

result_cols = st.columns(7)
result_cols[0].metric("Gross area Ag", f"{geometry.ag_mm2:,.0f} mm2")
result_cols[1].metric("phi", f"{phi:.2f}")
result_cols[2].metric("e_min", f"{manual_result.minimum_eccentricity_mm:.1f} mm")
result_cols[3].metric("Required Ast from PMM", required_ast_display)
result_cols[4].metric("Provided Ast", f"{ast_manual:,.1f} mm2")
result_cols[5].metric("Steel ratio", f"{manual_result.steel_ratio_percent:.3f} %")
result_cols[6].metric("Governing PMM ratio", f"{manual_result.governing_ratio:.3f}")

if mode == "Auto design":
    with st.spinner("Searching reinforcement layout..."):
        options_df = auto_design_options(
            geometry=geometry,
            cover_mm=cover_mm,
            tie_dia_mm=tie_dia_mm,
            fc_mpa=fc_mpa,
            fy_mpa=fy_mpa,
            phi=phi,
            column_type=column_type,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            min_clear_spacing_user_mm=min_clear_spacing_user_mm,
            shape=shape,
            bar_sizes_mm=available_sizes,
            load_df=load_cases_df,
            axial_cap_factor_value=axial_cap,
        )
else:
    options_df = pd.DataFrame()

tabs = st.tabs(["Results", "Section", "Interaction", "Verification", "Method"])

with tabs[0]:
    status_text = "OK" if manual_result.overall_ok else "NG"
    st.subheader("Manual Check Summary")
    summary_df = pd.DataFrame(
        [
            ["Status", status_text, ""],
            ["Governing case", manual_result.governing_load_case, ""],
            ["PMM ratio", f"{manual_result.governing_ratio:,.3f}", "<= 1.0 required"],
            ["Spacing check", "Pass" if manual_result.spacing_ok else "Fail", manual_result.spacing_note],
            ["Required Ast from PMM", required_ast_display, "Equivalent steel area for the current layout pattern"],
            ["PMM requirement status", "Solved within rho_max" if manual_result.pmm_required_found else "Exceeds rho_max", ""],
            ["Axial starter Ast", f"{manual_result.ast_axial_starter_mm2:,.1f}", "mm2"],
            ["Ast from equation", f"{manual_result.ast_from_equation_mm2:,.1f}", "mm2"],
            ["Ast minimum", f"{manual_result.ast_min_mm2:,.1f}", "mm2"],
            ["Ast maximum", f"{manual_result.ast_max_mm2:,.1f}", "mm2"],
            ["Ast provided", f"{ast_manual:,.1f}", "mm2"],
            ["Minimum eccentricity", f"{manual_result.minimum_eccentricity_mm:,.1f}", "mm"],
            ["Minimum moment floor at Pu", f"{governing_case['m_min']:,.1f}", "kN-m"],
            ["Mx input governing", f"{governing_case['mx_input']:,.1f}", "kN-m"],
            ["My input governing", f"{governing_case['my_input']:,.1f}", "kN-m"],
            ["Mx used in PMM", f"{governing_case['mx']:,.1f}", "kN-m"],
            ["My used in PMM", f"{governing_case['my']:,.1f}", "kN-m"],
            ["Minimum eccentricity applied", "Yes" if governing_case["min_ecc_applied"] else "No", ""],
            ["phi Mnx at governing Pu", f"{manual_result.phi_mnx_at_pu_kNm:,.1f}", "kN-m"],
            ["phi Mny at governing Pu", f"{manual_result.phi_mny_at_pu_kNm:,.1f}", "kN-m"],
            ["phi Pn", f"{manual_result.phi_pn_kN:,.1f}", "kN"],
            ["Axial cap factor", f"{axial_cap:.2f}", "ACI compression limit"],
            ["Pu governing", f"{governing_case['pu']:,.1f}", "kN"],
        ],
        columns=["Item", "Value", "Unit / Note"],
    )
    st.dataframe(summary_df, use_container_width=True, hide_index=True)
    st.subheader("Internal Force Cases")
    st.dataframe(load_case_results_df, use_container_width=True, hide_index=True)

    if mode == "Auto design":
        st.subheader("Auto Design Options")
        if options_df.empty:
            st.warning("No feasible layout found from the selected bar sizes and spacing limits.")
        else:
            st.dataframe(options_df, use_container_width=True, hide_index=True)
            best = options_df.iloc[0]
            st.info(
                f"Suggested starting option: {int(best['Total bars'])} bars, DB{int(best['Bar size (mm)'])}, "
                f"{best['Arrangement']}, Ast = {best['Ast provided (mm2)']:,.1f} mm2"
            )

with tabs[1]:
    top_left, top_right = st.columns(2)
    with top_left:
        st.plotly_chart(section_layout_fig, use_container_width=True)
    with top_right:
        st.plotly_chart(uniaxial_fig, use_container_width=True)

    st.subheader("Minimum Reinforcement Check")
    min_check_df = pd.DataFrame(
        [
            ["Minimum Ast required", f"{manual_result.ast_min_mm2:,.1f} mm2", "From minimum rho setting"],
            ["Provided Ast", f"{ast_manual:,.1f} mm2", "Current layout"],
            ["Overall status", "OK" if ast_manual >= manual_result.ast_min_mm2 else "NG", "Minimum reinforcement check"],
        ],
        columns=["Check", "Value", "Note"],
    )
    st.dataframe(min_check_df, use_container_width=True, hide_index=True)
    st.plotly_chart(pmm_slice_fig, use_container_width=True)
    st.plotly_chart(pmm_surface_fig, use_container_width=True, key="pmm_3d_surface")

with tabs[2]:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(axial_capacity_plot(manual_result, governing_case["pu"]), use_container_width=True)
    with right:
        st.plotly_chart(interaction_plot(curve_x_manual, curve_y_manual, governing_case["pu"]), use_container_width=True)
        st.dataframe(
            pd.DataFrame(
                [
                    ["Max phi Pn from x-curve", f"{curve_x_manual['phi_pn_kN'].max():,.1f}", "kN"],
                    ["phi Mnx at governing Pu", f"{manual_result.phi_mnx_at_pu_kNm:,.1f}", "kN-m"],
                    ["Max phi Pn from y-curve", f"{curve_y_manual['phi_pn_kN'].max():,.1f}", "kN"],
                    ["phi Mny at governing Pu", f"{manual_result.phi_mny_at_pu_kNm:,.1f}", "kN-m"],
                ],
                columns=["Interaction item", "Value", "Unit"],
            ),
            use_container_width=True,
            hide_index=True,
        )

with tabs[3]:
    st.subheader("Verification Benchmarks")
    st.dataframe(verification_df, use_container_width=True, hide_index=True)
    st.caption("These are built-in sanity checks for square, rectangular, and circular benchmark sections. They help confirm symmetry, intercept consistency, and rotational behavior, but they do not replace an external hand-check or certified design software benchmark.")

with tabs[4]:
    st.markdown(
        f"""
        **Workflow intentionally modeled after the reference app**

        - Sidebar-first input flow with grouped sections for `Design Basis`, `Geometry`, `Material and Load`, and `Reinforcement`
        - Editable table for internal force load cases: `Pu`, `Mx`, `My`
        - `Auto design` and `Manual check` modes
        - Save/load project state with JSON
        - Result metrics, summary tables, dedicated `Section` and `Interaction` tabs, and transparent calculation notes

        **Current design basis**

        - Main section checks use `strain compatibility PMM`, not a simplified load contour
        - ACI axial compression cap is still enforced on the compression end:
          `phi Pn,max = cap x phi x [0.85 f'c (Ag - Ast) + fy Ast]`
        - `cap = 0.80` for tied columns and `0.85` for spiral columns
        - The app still computes an axial starter steel estimate from:
          `Ast = (Pu/(cap x phi) - 0.85 f'c Ag) / (fy - 0.85 f'c)`
        - The axial starter is used only as a screening and reporting value:
          `Ast starter = max(Ast from equation, Ast minimum, 0)`
        - Steel ratio limits:
          `rho_min = {min_ratio_percent:.2f}%`
          `rho_max = {max_ratio_percent:.2f}%`

        **Layout logic**

        - Rectangular sections use perimeter-bar style input:
          `bars on top/bottom` and `bars on left/right`
        - Total bars for rectangular sections:
          `n = 2 * bars_width_face + 2 * (bars_depth_face - 2)`
        - Circular sections place bars uniformly around the perimeter
        - Minimum spacing check uses the largest of user input, `1.5db`, and `40 mm`

        **Strain compatibility interaction engine**

        - Square, rectangular, and circular sections all generate uniaxial `P-Mx` and `P-My` curves using strain compatibility
        - Concrete compression is integrated over the compression zone with `0.85f'c`
        - Steel stress is computed from strain using `Es = 200,000 MPa` and capped at `fy`
        - Bars inside the compression block use net steel stress `(fs - 0.85f'c)` to avoid double counting displaced concrete
        - Uniaxial curves use strain-based `phi` transition between compression-controlled and tension-controlled behavior
        - The compression end of the uniaxial curve is capped to the ACI axial compression limit for tied/spiral columns

        **Biaxial bending check**

        - Every load case in the input table is checked against the true rotated-neutral-axis PMM surface
        - At each `Pu`, the app also interpolates `phi Mnx(Pu)` and `phi Mny(Pu)` for reference reporting
        - Minimum eccentricity is enforced as `Mmin = Pu x e_min`
        - `e_min = 0.1D` for circular sections and `0.1 x min(b, h)` for square/rectangular sections in the current implementation
        - If user-entered `Mx, My` are smaller than `Mmin`, the app scales the moment vector up to the minimum required magnitude before the PMM check
        - The worst adjusted load case becomes the governing case for summary and auto design
        - The `Section` tab now shows:
          base reinforcement plot, uniaxial interaction curves with demand markers, `PMM Mux-My` slice at governing `Pu`, and a 3D interaction surface with the load point

        **Required steel reporting**

        - `Required Ast from PMM` is solved iteratively for the current layout pattern by increasing or reducing the equivalent total steel area until the governing PMM ratio reaches `1.0`
        - This PMM-required steel value is an equivalent area for the current bar pattern and bar centroids
        - `Axial starter Ast` remains visible as a quick screening value, but it is no longer the pass/fail criterion

        **Engineering note**

        - App status is now governed by spacing, steel ratio bounds, and true PMM utilization, instead of mixing axial-only and PMM-only criteria
        - The summary governing ratio, the `Section` tab PMM slice, the 3D surface, and auto-design screening all use the same PMM engine
        - `phi Pn` remains useful as a compression reference value, but it is no longer treated as a separate duplicate pass/fail gate
        - Final design should still verify the governing code, moment effects, slenderness, confinement, pile driving or precast detailing limits, splice/development, and project-specific requirements
        """
    )
