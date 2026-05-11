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
    ast_from_equation_mm2: float
    ast_min_mm2: float
    ast_max_mm2: float
    phi_pn_kN: float
    steel_ratio_percent: float
    spacing_ok: bool
    spacing_note: str
    overall_ok: bool
    eccentricity_mm: float
    governing_ratio: float = 0.0
    governing_load_case: str = ""
    phi_mnx_at_pu_kNm: float = 0.0
    phi_mny_at_pu_kNm: float = 0.0


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


def design_strength_kN(phi: float, fc_mpa: float, fy_mpa: float, ag_mm2: float, ast_mm2: float) -> float:
    nominal_n = 0.85 * fc_mpa * (ag_mm2 - ast_mm2) + fy_mpa * ast_mm2
    return phi * nominal_n / 1000.0


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
) -> tuple[float, float, float]:
    pu_n = pu_kN * 1000.0
    ast_min = min_ratio * ag_mm2
    numerator = pu_n / phi - 0.85 * fc_mpa * ag_mm2
    denominator = fy_mpa - 0.85 * fc_mpa
    ast_equation = numerator / denominator if denominator > 0 else 0.0
    ast_required = max(ast_min, ast_equation, 0.0)
    return ast_required, ast_equation, ast_min


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

    width_clear = float("inf") if bars_width_face <= 1 else (core_width - bar_dia_mm) / (bars_width_face - 1) - bar_dia_mm
    depth_clear = float("inf") if bars_depth_face <= 1 else (core_depth - bar_dia_mm) / (bars_depth_face - 1) - bar_dia_mm
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
    return nominal_p_n, abs(nominal_m_nmm), max_tension_strain


def interaction_curve(
    geometry: SectionGeometry,
    shape: Shape,
    cover_mm: float,
    tie_dia_mm: float,
    bar_dia_mm: float,
    fc_mpa: float,
    fy_mpa: float,
    phi: float,
    axis: Literal["x", "y"],
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
    bar_area_single = bar_area_mm2(bar_dia_mm)
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
                phi_pn_kN=phi * nominal_p_n / 1000.0,
                phi_mn_kNm=phi * nominal_m_nmm / 1_000_000.0,
                neutral_axis_mm=c,
                max_tension_strain=max_tension_strain,
            )
        )

    df = pd.DataFrame([point.__dict__ for point in rows])
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


def biaxial_ratio(mx_kNm: float, my_kNm: float, mnx_kNm: float, mny_kNm: float, alpha: float) -> float:
    if mnx_kNm <= 0.0 or mny_kNm <= 0.0:
        return float("inf")
    return (abs(mx_kNm) / mnx_kNm) ** alpha + (abs(my_kNm) / mny_kNm) ** alpha


def evaluate_load_cases(load_df: pd.DataFrame, curve_x: pd.DataFrame, curve_y: pd.DataFrame, alpha: float) -> tuple[pd.DataFrame, dict]:
    rows: list[dict] = []
    governing: dict | None = None

    for _, row in load_df.iterrows():
        case = str(row["Case"])
        pu = float(row["Pu (kN)"])
        mx = float(row["Mx (kN-m)"])
        my = float(row["My (kN-m)"])
        mnx = moment_capacity_at_pu(curve_x, pu)
        mny = moment_capacity_at_pu(curve_y, pu)
        ratio = biaxial_ratio(mx, my, mnx, mny, alpha)
        status = "OK" if ratio <= 1.0 else "NG"
        item = {
            "Case": case,
            "Pu (kN)": round(pu, 1),
            "Mx (kN-m)": round(mx, 1),
            "My (kN-m)": round(my, 1),
            "phi Mnx at Pu (kN-m)": round(mnx, 1),
            "phi Mny at Pu (kN-m)": round(mny, 1),
            "Biaxial ratio": round(ratio, 3) if math.isfinite(ratio) else None,
            "Status": status,
        }
        rows.append(item)
        if governing is None or ratio > governing["ratio"]:
            governing = {"case": case, "ratio": ratio, "mnx": mnx, "mny": mny, "pu": pu, "mx": mx, "my": my}

    return pd.DataFrame(rows), (governing or {"case": "", "ratio": 0.0, "mnx": 0.0, "mny": 0.0, "pu": 0.0, "mx": 0.0, "my": 0.0})


def analyze_manual_layout(
    geometry: SectionGeometry,
    cover_mm: float,
    tie_dia_mm: float,
    bar_dia_mm: float,
    min_clear_spacing_user_mm: float,
    phi: float,
    fc_mpa: float,
    fy_mpa: float,
    pu_kN: float,
    min_ratio: float,
    max_ratio: float,
    shape: Shape,
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
    ast_required, ast_equation, ast_min = required_steel_area_mm2(governing_pu_kN, phi, fc_mpa, fy_mpa, geometry.ag_mm2, min_ratio)
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
        eccentricity = 0.0
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
        eccentricity = 0.1 * min(float(geometry.width_mm), float(geometry.depth_mm))

    phi_pn = design_strength_kN(phi, fc_mpa, fy_mpa, geometry.ag_mm2, ast)
    overall_ok = spacing_ok and ast >= ast_required and ast <= ast_max and governing_ratio <= 1.0 and phi_pn >= governing_pu_kN
    return DesignResult(
        ast_required_mm2=ast_required,
        ast_from_equation_mm2=ast_equation,
        ast_min_mm2=ast_min,
        ast_max_mm2=ast_max,
        phi_pn_kN=phi_pn,
        steel_ratio_percent=100.0 * ast / geometry.ag_mm2,
        spacing_ok=spacing_ok,
        spacing_note=spacing_note,
        overall_ok=overall_ok,
        eccentricity_mm=eccentricity,
        governing_ratio=governing_ratio,
        governing_load_case=governing_case_name,
        phi_mnx_at_pu_kNm=phi_mnx_at_pu_kNm,
        phi_mny_at_pu_kNm=phi_mny_at_pu_kNm,
    )


def auto_design_options(
    geometry: SectionGeometry,
    cover_mm: float,
    tie_dia_mm: float,
    fc_mpa: float,
    fy_mpa: float,
    phi: float,
    min_ratio: float,
    max_ratio: float,
    min_clear_spacing_user_mm: float,
    shape: Shape,
    bar_sizes_mm: list[float],
    load_df: pd.DataFrame,
    biaxial_alpha: float,
) -> pd.DataFrame:
    rows: list[dict] = []

    for dia in bar_sizes_mm:
        if shape == "Circular":
            for n_bars in range(6, 33):
                curve_x = interaction_curve(geometry, shape, cover_mm, tie_dia_mm, dia, fc_mpa, fy_mpa, phi, "x", bars_circular=n_bars)
                curve_y = interaction_curve(geometry, shape, cover_mm, tie_dia_mm, dia, fc_mpa, fy_mpa, phi, "y", bars_circular=n_bars)
                _, governing = evaluate_load_cases(load_df, curve_x, curve_y, biaxial_alpha)
                result = analyze_manual_layout(
                    geometry=geometry,
                    cover_mm=cover_mm,
                    tie_dia_mm=tie_dia_mm,
                    bar_dia_mm=dia,
                    min_clear_spacing_user_mm=min_clear_spacing_user_mm,
                    phi=phi,
                    fc_mpa=fc_mpa,
                    fy_mpa=fy_mpa,
                    pu_kN=governing["pu"],
                    min_ratio=min_ratio,
                    max_ratio=max_ratio,
                    shape=shape,
                    governing_pu_kN=governing["pu"],
                    governing_case_name=governing["case"],
                    governing_ratio=governing["ratio"],
                    phi_mnx_at_pu_kNm=governing["mnx"],
                    phi_mny_at_pu_kNm=governing["mny"],
                    bars_circular=n_bars,
                )
                ast = n_bars * bar_area_mm2(dia)
                if result.spacing_ok and ast <= result.ast_max_mm2 and result.overall_ok:
                    rows.append(
                        {
                            "Bar size (mm)": dia,
                            "Arrangement": f"{n_bars} bars around perimeter",
                            "Total bars": n_bars,
                            "Ast provided (mm2)": round(ast, 1),
                            "Steel ratio (%)": round(result.steel_ratio_percent, 3),
                            "Governing case": governing["case"],
                            "Biaxial ratio": round(governing["ratio"], 3),
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
                    curve_x = interaction_curve(geometry, shape, cover_mm, tie_dia_mm, dia, fc_mpa, fy_mpa, phi, "x", bars_width_face=bx, bars_depth_face=by)
                    curve_y = interaction_curve(geometry, shape, cover_mm, tie_dia_mm, dia, fc_mpa, fy_mpa, phi, "y", bars_width_face=bx, bars_depth_face=by)
                    _, governing = evaluate_load_cases(load_df, curve_x, curve_y, biaxial_alpha)
                    result = analyze_manual_layout(
                        geometry=geometry,
                        cover_mm=cover_mm,
                        tie_dia_mm=tie_dia_mm,
                        bar_dia_mm=dia,
                        min_clear_spacing_user_mm=min_clear_spacing_user_mm,
                        phi=phi,
                        fc_mpa=fc_mpa,
                        fy_mpa=fy_mpa,
                        pu_kN=governing["pu"],
                        min_ratio=min_ratio,
                        max_ratio=max_ratio,
                        shape=shape,
                        governing_pu_kN=governing["pu"],
                        governing_case_name=governing["case"],
                        governing_ratio=governing["ratio"],
                        phi_mnx_at_pu_kNm=governing["mnx"],
                        phi_mny_at_pu_kNm=governing["mny"],
                        bars_width_face=bx,
                        bars_depth_face=by,
                    )
                    ast = total * bar_area_mm2(dia)
                    if result.spacing_ok and ast <= result.ast_max_mm2 and result.overall_ok:
                        rows.append(
                            {
                                "Bar size (mm)": dia,
                                "Arrangement": f"{bx} bars on top/bottom, {by} bars on left/right",
                                "Total bars": total,
                                "Ast provided (mm2)": round(ast, 1),
                                "Steel ratio (%)": round(result.steel_ratio_percent, 3),
                                "Governing case": governing["case"],
                                "Biaxial ratio": round(governing["ratio"], 3),
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
        title="Section Layout",
        template="plotly_white",
        height=520,
        xaxis={"title": "x (mm)", "scaleanchor": "y", "range": [-limit, limit]},
        yaxis={"title": "y (mm)", "range": [-limit, limit]},
        legend={"orientation": "h"},
        margin={"l": 20, "r": 20, "t": 50, "b": 20},
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


def project_payload(data: dict) -> str:
    return json.dumps(data, indent=2)


st.set_page_config(page_title="Pile Reinforcement Designer", page_icon="P", layout="wide", initial_sidebar_state="expanded")

st.title("Pile Reinforcement Designer")
st.caption("Preliminary RC pile longitudinal reinforcement design for square, rectangular, and circular sections with auto design, manual check, layout plots, and save/load workflow.")

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
        "biaxial_alpha": 1.0,
        "mode": "Auto design",
        "manual_bar_dia_mm": 20.0,
        "bars_width_face": 4,
        "bars_depth_face": 4,
        "bars_circular": 8,
        "available_sizes_text": "12, 16, 20, 25, 28, 32",
    }
    payload = project_payload({key: st.session_state.get(key, value) for key, value in default_state.items()})
    st.download_button("Download project JSON", payload, file_name="pile_rebar_design.json", mime="application/json", use_container_width=True)
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
    code_basis = st.selectbox("Reference basis", ["ACI-style simplified axial compression"], index=0)
    column_type = st.radio("Transverse reinforcement type", ["Tied", "Spiral"], horizontal=True, key="column_type")
    min_ratio_percent = st.number_input("Minimum rho for auto (%)", min_value=0.1, max_value=8.0, step=0.1, key="min_ratio_percent")
    max_ratio_percent = st.number_input("Maximum rho for auto (%)", min_value=1.0, max_value=12.0, step=0.1, key="max_ratio_percent")
    biaxial_alpha = st.number_input("Biaxial alpha", min_value=1.0, max_value=2.0, step=0.1, key="biaxial_alpha")
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
min_ratio = min_ratio_percent / 100.0
max_ratio = max_ratio_percent / 100.0

available_sizes = sorted(
    {
        float(item.strip())
        for item in available_sizes_text.split(",")
        if item.strip()
    }
)
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
    axis="x",
    bars_width_face=int(bars_width_face) if bars_width_face is not None else None,
    bars_depth_face=int(bars_depth_face) if bars_depth_face is not None else None,
    bars_circular=int(bars_circular) if bars_circular is not None else None,
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
    axis="y",
    bars_width_face=int(bars_width_face) if bars_width_face is not None else None,
    bars_depth_face=int(bars_depth_face) if bars_depth_face is not None else None,
    bars_circular=int(bars_circular) if bars_circular is not None else None,
)
load_case_results_df, governing_case = evaluate_load_cases(load_cases_df, curve_x_manual, curve_y_manual, biaxial_alpha)

manual_result = analyze_manual_layout(
    geometry=geometry,
    cover_mm=cover_mm,
    tie_dia_mm=tie_dia_mm,
    bar_dia_mm=float(manual_bar_dia_mm),
    min_clear_spacing_user_mm=min_clear_spacing_user_mm,
    phi=phi,
    fc_mpa=fc_mpa,
    fy_mpa=fy_mpa,
    pu_kN=governing_case["pu"],
    min_ratio=min_ratio,
    max_ratio=max_ratio,
    shape=shape,
    governing_pu_kN=governing_case["pu"],
    governing_case_name=governing_case["case"],
    governing_ratio=governing_case["ratio"],
    phi_mnx_at_pu_kNm=governing_case["mnx"],
    phi_mny_at_pu_kNm=governing_case["mny"],
    bars_width_face=int(bars_width_face) if bars_width_face is not None else None,
    bars_depth_face=int(bars_depth_face) if bars_depth_face is not None else None,
    bars_circular=int(bars_circular) if bars_circular is not None else None,
)

if shape == "Circular":
    total_bars = int(bars_circular)
else:
    total_bars = total_bars_rectangular(int(bars_width_face), int(bars_depth_face))
ast_manual = total_bars * bar_area_mm2(float(manual_bar_dia_mm))

result_cols = st.columns(6)
result_cols[0].metric("Gross area Ag", f"{geometry.ag_mm2:,.0f} mm2")
result_cols[1].metric("phi", f"{phi:.2f}")
result_cols[2].metric("Required Ast", f"{manual_result.ast_required_mm2:,.1f} mm2")
result_cols[3].metric("Provided Ast", f"{ast_manual:,.1f} mm2")
result_cols[4].metric("Steel ratio", f"{manual_result.steel_ratio_percent:.3f} %")
result_cols[5].metric("Governing ratio", f"{manual_result.governing_ratio:.3f}")

if mode == "Auto design":
    with st.spinner("Searching reinforcement layout..."):
        options_df = auto_design_options(
            geometry=geometry,
            cover_mm=cover_mm,
            tie_dia_mm=tie_dia_mm,
            fc_mpa=fc_mpa,
            fy_mpa=fy_mpa,
            phi=phi,
            min_ratio=min_ratio,
            max_ratio=max_ratio,
            min_clear_spacing_user_mm=min_clear_spacing_user_mm,
            shape=shape,
            bar_sizes_mm=available_sizes,
            load_df=load_cases_df,
            biaxial_alpha=biaxial_alpha,
        )
else:
    options_df = pd.DataFrame()

tabs = st.tabs(["Results", "Section", "Interaction", "Method"])

with tabs[0]:
    status_text = "OK" if manual_result.overall_ok else "NG"
    st.subheader("Manual Check Summary")
    summary_df = pd.DataFrame(
        [
            ["Status", status_text, ""],
            ["Governing case", manual_result.governing_load_case, ""],
            ["Biaxial ratio", f"{manual_result.governing_ratio:,.3f}", "<= 1.0 required"],
            ["Spacing check", "Pass" if manual_result.spacing_ok else "Fail", manual_result.spacing_note],
            ["Ast from equation", f"{manual_result.ast_from_equation_mm2:,.1f}", "mm2"],
            ["Ast minimum", f"{manual_result.ast_min_mm2:,.1f}", "mm2"],
            ["Ast maximum", f"{manual_result.ast_max_mm2:,.1f}", "mm2"],
            ["Ast provided", f"{ast_manual:,.1f}", "mm2"],
            ["phi Mnx at governing Pu", f"{manual_result.phi_mnx_at_pu_kNm:,.1f}", "kN-m"],
            ["phi Mny at governing Pu", f"{manual_result.phi_mny_at_pu_kNm:,.1f}", "kN-m"],
            ["phi Pn", f"{manual_result.phi_pn_kN:,.1f}", "kN"],
            ["Pu governing", f"{governing_case['pu']:,.1f}", "kN"],
            ["Mx governing", f"{governing_case['mx']:,.1f}", "kN-m"],
            ["My governing", f"{governing_case['my']:,.1f}", "kN-m"],
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
    st.plotly_chart(
        section_figure(
            geometry=geometry,
            shape=shape,
            cover_mm=cover_mm,
            tie_dia_mm=tie_dia_mm,
            bar_dia_mm=float(manual_bar_dia_mm),
            bars_width_face=int(bars_width_face) if bars_width_face is not None else None,
            bars_depth_face=int(bars_depth_face) if bars_depth_face is not None else None,
            bars_circular=int(bars_circular) if bars_circular is not None else None,
        ),
        use_container_width=True,
    )
    st.caption("The plot mirrors the reference app style by showing section outline, tie line, and actual main-bar placement.")

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
    st.markdown(
        f"""
        **Workflow intentionally modeled after the reference app**

        - Sidebar-first input flow with grouped sections for `Design Basis`, `Geometry`, `Material and Load`, and `Reinforcement`
        - Editable table for internal force load cases: `Pu`, `Mx`, `My`
        - `Auto design` and `Manual check` modes
        - Save/load project state with JSON
        - Result metrics, summary tables, dedicated `Section` and `Interaction` tabs, and transparent calculation notes

        **Current design basis**

        - Simplified concentric compression check:
          `phi Pn = phi [0.85 f'c (Ag - Ast) + fy Ast]`
        - Required steel from governing axial load case:
          `Ast = (Pu/phi - 0.85 f'c Ag) / (fy - 0.85 f'c)`
        - Adopted `Ast required = max(Ast from equation, Ast minimum, 0)`
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

        - Both rectangular and circular sections now generate uniaxial `P-Mx` and `P-My` interaction curves using strain compatibility
        - Concrete compression is integrated over the compression zone with `0.85f'c`
        - Steel stress is computed from strain using `Es = 200,000 MPa` and capped at `fy`
        - Bars inside the compression block use net steel stress `(fs - 0.85f'c)` to avoid double counting displaced concrete
        - Neutral axis depth is swept through multiple values to build the full interaction curve

        **Biaxial bending check**

        - The app evaluates every load case in the input table
        - At each `Pu`, it interpolates `phi Mnx(Pu)` and `phi Mny(Pu)` from the interaction curves
        - Biaxial utilization is checked with load contour form:
          `(Mx / phi Mnx)^alpha + (My / phi Mny)^alpha <= 1.0`
        - `alpha = {biaxial_alpha:.2f}` in the current run
        - The worst load case becomes the governing case for the manual summary and for auto design filtering

        **Engineering note**

        - This app is now much closer in behavior and presentation to your reference app, and the interaction plots now come from a true strain compatibility workflow for both rectangular and circular sections
        - Final design should still verify the governing code, moment effects, slenderness, confinement, pile driving or precast detailing limits, splice/development, and project-specific requirements
        """
    )
