import json
import importlib
import math
import streamlit as st
import pandas as pd

# Streamlit reruns app.py in the same Python process. Reload the local
# calculation module so newly-added presets are visible without a server restart.
import stm_calculations as _stm_calculations
_stm_calculations = importlib.reload(_stm_calculations)

from stm_calculations import (
    get_preset_layouts, get_truncated_triangle_equal,
    validate_pile_spacing, compute_cap_bounds_per_side,
    parse_custom_coords, stm_design,
    check_rebar, suggest_rebar, REBAR_DB,
    REBAR_FY, REBAR_DIAM_MM, FY_CAP_MPA,
    compute_pile_reactions,
    optimize_rebar, check_anchorage,
    compute_top_reinforcement,
)
def _cap_area_m2(cap_polygon, cap_lx, cap_ly):
    """คำนวณพื้นที่ฐานราก (m²)
    - มี cap_polygon (truncated triangle) → Shoelace formula (พื้นที่จริง)
    - ไม่มี cap_polygon → Lx × Ly (bounding box)"""
    if cap_polygon and len(cap_polygon) >= 3:
        pts = cap_polygon
        n = len(pts)
        area = abs(sum(pts[i][0]*pts[(i+1)%n][1] -
                       pts[(i+1)%n][0]*pts[i][1]
                       for i in range(n))) / 2.0
        return area / 1e6  # mm² → m²
    return (cap_lx / 1000.0) * (cap_ly / 1000.0)


def _formula_box(text):
    """Render formula text without Streamlit's dynamic syntax highlighter."""
    safe = (str(text)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;"))
    st.markdown(
        (
            '<div style="font-family: ui-monospace, SFMono-Regular, Menlo, '
            'Consolas, Liberation Mono, monospace; white-space: pre-wrap; '
            'background:#f8fafc; border:1px solid #e2e8f0; '
            'border-radius:6px; padding:0.65rem 0.75rem; '
            'color:#0f172a; font-size:0.92rem; line-height:1.45;">'
            '{}</div>'
        ).format(safe),
        unsafe_allow_html=True)


def _rebar_spacing_text(n_bars, distribution_width_mm, cap_lx_mm, cap_ly_mm,
                        cover_mm):
    """Estimate center-to-center spacing used for plan rebar layout."""
    n = int(n_bars)
    if n <= 1:
        return "—"
    edge_inset = min(max(float(cover_mm), 25.0),
                     0.45 * min(float(cap_lx_mm), float(cap_ly_mm)))
    usable_width = max(0.0, float(distribution_width_mm) - 2.0 * edge_inset)
    return "{:.0f}".format(usable_width / (n - 1))


def _ensure_top_rebar_schedule(tr, cap_lx_mm, cap_ly_mm, cover_mm):
    """Backfill top-bar detailing schedule if result dict lacks it.

    This keeps the UI/report robust when a rerun has old session data or a
    deployment briefly serves mixed app/calculation modules.
    """
    bar = tr.get("top_bar_size", "DB20")
    A_bar = REBAR_DB.get(bar, REBAR_DB["DB20"])
    db = tr.get("db_top_mm", REBAR_DIAM_MM.get(bar, 20.0))
    s_max = float(tr.get("s_max_top_mm", min(3.0 * float(cap_ly_mm), 450.0)))
    edge_inset = tr.get("top_edge_inset_mm")
    if edge_inset is None:
        edge_inset = min(max(float(cover_mm), float(db)),
                         0.45 * min(float(cap_lx_mm), float(cap_ly_mm)))
    usable_x = max(0.0, float(cap_lx_mm) - 2.0 * edge_inset)
    usable_y = max(0.0, float(cap_ly_mm) - 2.0 * edge_inset)

    def _schedule(As_req, distribution_width):
        n_area = max(2, int(math.ceil(As_req / A_bar))) if As_req > 0 else 2
        n_spacing = (
            max(2, int(math.ceil(distribution_width / s_max)) + 1)
            if s_max > 0 else n_area)
        n = max(n_area, n_spacing)
        spacing = distribution_width / (n - 1) if n > 1 else 0.0
        return n, spacing, n * A_bar, n_area, n_spacing

    x_n, x_spacing, x_As, x_n_area, x_n_spacing = _schedule(
        float(tr.get("As_top_x_mm2", 0.0)), usable_y)
    y_n, y_spacing, y_As, y_n_area, y_n_spacing = _schedule(
        float(tr.get("As_top_y_mm2", 0.0)), usable_x)
    x_valid = (
        int(tr.get("top_x_n_bars") or 0) >= 2 and
        float(tr.get("top_x_spacing_mm") or 0.0) > 0.0 and
        float(tr.get("top_x_As_provided_mm2") or 0.0) > 0.0
    )
    y_valid = (
        int(tr.get("top_y_n_bars") or 0) >= 2 and
        float(tr.get("top_y_spacing_mm") or 0.0) > 0.0 and
        float(tr.get("top_y_As_provided_mm2") or 0.0) > 0.0
    )

    tr.update({
        "top_bar_area_mm2": A_bar,
        "top_edge_inset_mm": edge_inset,
        "top_usable_x_mm": usable_x,
        "top_usable_y_mm": usable_y,
        "top_x_n_bars": int(tr.get("top_x_n_bars") if x_valid else x_n),
        "top_y_n_bars": int(tr.get("top_y_n_bars") if y_valid else y_n),
        "top_x_spacing_mm": float(tr.get("top_x_spacing_mm") if x_valid else x_spacing),
        "top_y_spacing_mm": float(tr.get("top_y_spacing_mm") if y_valid else y_spacing),
        "top_x_As_provided_mm2": float(tr.get("top_x_As_provided_mm2") if x_valid else x_As),
        "top_y_As_provided_mm2": float(tr.get("top_y_As_provided_mm2") if y_valid else y_As),
        "top_x_n_area": int(tr.get("top_x_n_area") if x_valid else x_n_area),
        "top_y_n_area": int(tr.get("top_y_n_area") if y_valid else y_n_area),
        "top_x_n_spacing": int(tr.get("top_x_n_spacing") if x_valid else x_n_spacing),
        "top_y_n_spacing": int(tr.get("top_y_n_spacing") if y_valid else y_n_spacing),
    })
    return tr


def _check_anchorage_with_mode(bar_size, fc, fy_mpa,
                               available_straight_mm,
                               available_edge_hook_mm,
                               mode, available_vertical_hook_mm):
    """Call check_anchorage without new kwargs for deployment compatibility."""
    use_vertical = str(mode).startswith("90")
    hook_avail = (
        available_vertical_hook_mm if use_vertical
        else available_edge_hook_mm
    )
    out = check_anchorage(
        bar_size, fc, fy_mpa, available_straight_mm, hook_avail)
    out["anchorage_mode"] = mode
    out["available_edge_hook_mm"] = available_edge_hook_mm
    out["available_vertical_hook_mm"] = (
        available_vertical_hook_mm if use_vertical else None
    )
    out["available_hook_mm"] = hook_avail
    if use_vertical:
        if out.get("straight_ok"):
            out["recommended"] = "Straight bar OK"
        elif out.get("hook_ok"):
            out["recommended"] = "90° vertical hook OK"
        else:
            out["recommended"] = (
                "INSUFFICIENT — increase cap thickness or lower bottom bar"
            )
    return out


def _design_force_summary_rows(results):
    """Summarize the force demands actually used by the design checks."""
    rows = []
    struts = results.get("struts", [])
    if struts:
        gov_idx, gov_strut = max(
            enumerate(struts, 1),
            key=lambda item: item[1].get("F_strut_kN", 0.0))
        rows.append({
            "Component": "Strut compression",
            "Governing member / path": "S{} to P{}".format(gov_idx, gov_idx),
            "Design force (kN)": "{:.1f}".format(
                results.get("F_strut_max_kN",
                            gov_strut.get("F_strut_kN", 0.0))),
            "Used for": "Strut capacity DCR",
            "Notes": "max F_strut among all struts",
        })

    if results.get("is_3pile_resultant"):
        rows.append({
            "Component": "Tie resultant",
            "Governing member / path": "3-pile resultant",
            "Design force (kN)": "{:.1f}".format(
                results.get("F_tie_res_kN", 0.0)),
            "Used for": "Bottom X/Y As_STM",
            "Notes": "coordinate-rotation independent resultant",
        })
    else:
        rows.extend([
            {
                "Component": "Tie X direction",
                "Governing member / path": "max left/right tie demand",
                "Design force (kN)": "{:.1f}".format(
                    results.get("F_tie_x_design_kN",
                                results.get("F_tie_x_max_kN", 0.0))),
                "Used for": "Bottom X As_STM",
                "Notes": results.get("As_x_governs", ""),
            },
            {
                "Component": "Tie Y direction",
                "Governing member / path": "max front/back tie demand",
                "Design force (kN)": "{:.1f}".format(
                    results.get("F_tie_y_design_kN",
                                results.get("F_tie_y_max_kN", 0.0))),
                "Used for": "Bottom Y As_STM",
                "Notes": results.get("As_y_governs", ""),
            },
        ])
    return rows


def _strut_force_rows(results):
    """Return per-strut forces for the 3D-view force table."""
    rows = []
    for i, s in enumerate(results.get("struts", []), 1):
        node_x, node_y = s.get("column_coord", (0.0, 0.0))
        rows.append({
            "Strut": "S{}".format(i),
            "Pile": "P{}".format(i),
            "Top node X (mm)": "{:.0f}".format(node_x),
            "Top node Y (mm)": "{:.0f}".format(node_y),
            "P_i (kN)": "{:.1f}".format(s.get("P_i_kN", 0.0)),
            "F_strut (kN)": "{:.1f}".format(s.get("F_strut_kN", 0.0)),
            "θ (deg)": "{:.1f}".format(s.get("theta_deg", 0.0)),
            "Tie comp X (kN)": "{:.1f}".format(
                s.get("F_tie_x_kN", 0.0)),
            "Tie comp Y (kN)": "{:.1f}".format(
                s.get("F_tie_y_kN", 0.0)),
        })
    return rows


def _pile_head_force_rows(results):
    """Return pile-head actions transferred from the pile cap STM model."""
    data = []
    rows = []
    struts = results.get("struts", [])
    for i, s in enumerate(struts, 1):
        pile_x, pile_y = s.get("coord", (0.0, 0.0))
        node_x, node_y = s.get("column_coord", (0.0, 0.0))
        axial = float(s.get("P_i_kN", 0.0))
        hx_mag = float(s.get("F_tie_x_kN", 0.0))
        hy_mag = float(s.get("F_tie_y_kN", 0.0))
        dx = float(s.get("dx_from_col", pile_x - node_x))
        dy = float(s.get("dy_from_col", pile_y - node_y))
        hx = math.copysign(hx_mag, dx) if abs(dx) > 1e-9 else 0.0
        hy = math.copysign(hy_mag, dy) if abs(dy) > 1e-9 else 0.0
        h_res = math.hypot(hx, hy)
        strut_force = float(s.get("F_strut_kN", 0.0))
        theta = float(s.get("theta_deg", 0.0))
        note = "Compression pile"
        if axial < -1e-3:
            note = "Uplift/tension pile - strut ignored in STM"
        elif abs(axial) <= 1e-3:
            note = "Near-zero axial"
        item = {
            "idx": i,
            "pile": "P{}".format(i),
            "x": pile_x,
            "y": pile_y,
            "node_x": node_x,
            "node_y": node_y,
            "axial": axial,
            "hx": hx,
            "hy": hy,
            "h_res": h_res,
            "strut": strut_force,
            "theta": theta,
            "note": note,
        }
        data.append(item)
        rows.append({
            "Pile": item["pile"],
            "X (mm)": "{:.0f}".format(pile_x),
            "Y (mm)": "{:.0f}".format(pile_y),
            "Top node X (mm)": "{:.0f}".format(node_x),
            "Top node Y (mm)": "{:.0f}".format(node_y),
            "Axial P_i (kN)": "{:.1f}".format(axial),
            "H_x (kN)": "{:.1f}".format(hx),
            "H_y (kN)": "{:.1f}".format(hy),
            "H resultant (kN)": "{:.1f}".format(h_res),
            "F_strut (kN)": "{:.1f}".format(strut_force),
            "θ (deg)": "{:.1f}".format(theta),
            "Note": note,
        })

    summary = []
    if data:
        max_comp = max(data, key=lambda item: item["axial"])
        min_axial = min(data, key=lambda item: item["axial"])
        max_h = max(data, key=lambda item: item["h_res"])
        max_hx = max(data, key=lambda item: abs(item["hx"]))
        max_hy = max(data, key=lambda item: abs(item["hy"]))
        max_strut = max(data, key=lambda item: item["strut"])

        summary.append({
            "Critical action": "Max axial compression",
            "Pile": max_comp["pile"],
            "Design value": "{:.1f} kN".format(max_comp["axial"]),
            "Use for": "Pile axial compression design",
        })
        if min_axial["axial"] < -1e-3:
            summary.append({
                "Critical action": "Max uplift / tension",
                "Pile": min_axial["pile"],
                "Design value": "{:.1f} kN tension".format(
                    abs(min_axial["axial"])),
                "Use for": "Tension pile, anchorage, connection check",
            })
        summary.extend([
            {
                "Critical action": "Max horizontal resultant",
                "Pile": max_h["pile"],
                "Design value": "{:.1f} kN".format(max_h["h_res"]),
                "Use for": "Pile-head shear / dowel reinforcement",
            },
            {
                "Critical action": "Max |H_x|",
                "Pile": max_hx["pile"],
                "Design value": "{:.1f} kN".format(abs(max_hx["hx"])),
                "Use for": "Pile-head shear in X direction",
            },
            {
                "Critical action": "Max |H_y|",
                "Pile": max_hy["pile"],
                "Design value": "{:.1f} kN".format(abs(max_hy["hy"])),
                "Use for": "Pile-head shear in Y direction",
            },
            {
                "Critical action": "Max strut compression",
                "Pile": max_strut["pile"],
                "Design value": "{:.1f} kN".format(max_strut["strut"]),
                "Use for": "Pile-head compression bearing / STM node",
            },
        ])
    return rows, summary


def _signed_strut_component(struts, idx, axis):
    if idx >= len(struts):
        return 0.0
    s = struts[idx]
    if axis == "x":
        comp = abs(float(s.get("F_tie_x_kN", 0.0)))
        delta = float(s.get("dx_from_col", 0.0))
    else:
        comp = abs(float(s.get("F_tie_y_kN", 0.0)))
        delta = float(s.get("dy_from_col", 0.0))
    if abs(delta) <= 1e-9:
        return 0.0
    return math.copysign(comp, delta)


def _tie_member_rows(coords, results):
    """Return tie labels, endpoints, and estimated member forces.

    Forces are for the displayed tie paths and are estimated from cut
    equilibrium of the local strut horizontal components in the same row or
    column. The governing design forces remain the Tx/Ty summary above.
    """
    rows = []
    struts = results.get("struts", [])
    for tie_idx, (i, j) in enumerate(tie_pairs_for_3d_view(coords), 1):
        x1, y1 = coords[i]
        x2, y2 = coords[j]
        if abs(y1 - y2) <= 1.0:
            direction = "X tie path"
            cut = (x1 + x2) / 2.0
            row_ids = [
                k for k, (_x, _y) in enumerate(coords)
                if abs(_y - y1) <= 1.0
            ]
            left = sum(
                _signed_strut_component(struts, k, "x")
                for k in row_ids if coords[k][0] <= cut
            )
            right = sum(
                _signed_strut_component(struts, k, "x")
                for k in row_ids if coords[k][0] > cut
            )
            force = max(abs(left), abs(right))
            basis = "X cut at x={:.0f} mm".format(cut)
        elif abs(x1 - x2) <= 1.0:
            direction = "Y tie path"
            cut = (y1 + y2) / 2.0
            col_ids = [
                k for k, (_x, _y) in enumerate(coords)
                if abs(_x - x1) <= 1.0
            ]
            lower = sum(
                _signed_strut_component(struts, k, "y")
                for k in col_ids if coords[k][1] <= cut
            )
            upper = sum(
                _signed_strut_component(struts, k, "y")
                for k in col_ids if coords[k][1] > cut
            )
            force = max(abs(lower), abs(upper))
            basis = "Y cut at y={:.0f} mm".format(cut)
        else:
            direction = "Diagonal tie path"
            force = results.get("F_tie_res_kN", 0.0)
            basis = "3-pile resultant"
        rows.append({
            "Tie": "T{}".format(tie_idx),
            "From": "P{}".format(i + 1),
            "To": "P{}".format(j + 1),
            "Path": direction,
            "Tie force (kN)": "{:.1f}".format(force),
            "Basis": basis,
        })
    return rows


def _design_recommendations(results, x_chk=None, y_chk=None,
                            anch_x=None, anch_y=None,
                            opt_x=None, opt_y=None, cover=75.0):
    recs = []

    def add(issue, reason, action):
        recs.append({"Issue": issue, "Why": reason, "Recommended adjustment": action})

    if results.get("strut_DCR", 0.0) > 1.0:
        dcr = results["strut_DCR"]
        add(
            "Strut compression",
            "Strut DCR = {:.2f} > 1.00".format(dcr),
            "Increase pile size/strut bearing area or concrete strength; "
            "also consider increasing cap thickness or adding piles to reduce pile reactions.")

    if results.get("bearing_DCR", 0.0) > 1.0:
        dcr = results["bearing_DCR"]
        add(
            "Pile nodal bearing",
            "Bearing DCR = {:.2f} > 1.00".format(dcr),
            "Increase pile head area, concrete strength, or number of piles; "
            "move the column/load point closer to the pile-group centroid if eccentricity is high.")

    if results.get("column_DCR", 0.0) > 1.0:
        dcr = results["column_DCR"]
        add(
            "Column nodal bearing",
            "Column DCR = {:.2f} > 1.00".format(dcr),
            "Increase column size or concrete strength, or reduce factored column load.")

    if not results.get("angle_OK", True):
        bad = [s for s in results.get("struts", []) if s.get("theta_deg", 90.0) < 25.0]
        if bad:
            import math
            req_d = max(s["L_h"] * math.tan(math.radians(25.0)) for s in bad)
            req_h = req_d + cover + 12.5
            action = ("Increase cap thickness to about {:.0f} mm or more, "
                      "or reduce horizontal strut length by moving piles/column closer.").format(req_h)
        else:
            action = "Increase cap thickness or reduce pile-to-column horizontal distance."
        if results.get("stm_model_type") == "Point Column STM":
            action += (" For a long wall/abutment support, review whether "
                       "Wall/Abutment STM is the appropriate model before "
                       "increasing geometry solely to satisfy the angle.")
        add(
            "Strut angle",
            "Minimum strut angle = {:.1f} deg < 25 deg".format(
                results.get("min_strut_angle_deg", 0.0)),
            action)

    if results.get("has_uplift", False):
        add(
            "Uplift",
            "Minimum pile reaction P_min = {:.1f} kN".format(
                results.get("P_min_kN", 0.0)),
            "Provide tension pile/anchorage, add piles on the uplift side, "
            "reduce moment/eccentricity, or move the column/load point closer to the pile-group centroid.")

    if not results.get("reaction_equilibrium_OK", True):
        add(
            "Reaction equilibrium",
            "The pile layout cannot resist the applied load/moment with available lever arms.",
            "Add pile rows/columns in the missing lever-arm direction, revise custom coordinates, "
            "or move the column/load point back inside the effective pile group.")

    if x_chk is not None and not x_chk.get("ok", True):
        if opt_x:
            action = "Use at least {}-{} in X direction, or increase bar count/size.".format(
                opt_x["n_bars"], opt_x["bar_size"])
        else:
            action = "Increase X-direction bar count or select a larger bar size."
        add(
            "Bottom tie steel X",
            "Provided As ratio = {:.2f}; governing demand = {}".format(
                x_chk.get("ratio", 0.0), results.get("As_x_governs", "—")),
            action)

    if y_chk is not None and not y_chk.get("ok", True):
        if opt_y:
            action = "Use at least {}-{} in Y direction, or increase bar count/size.".format(
                opt_y["n_bars"], opt_y["bar_size"])
        else:
            action = "Increase Y-direction bar count or select a larger bar size."
        add(
            "Bottom tie steel Y",
            "Provided As ratio = {:.2f}; governing demand = {}".format(
                y_chk.get("ratio", 0.0), results.get("As_y_governs", "—")),
            action)

    if anch_x is not None and not anch_x.get("ok", True):
        if str(anch_x.get("anchorage_mode", "")).startswith("90"):
            action = ("Increase cap thickness, lower the bottom-bar layer if "
                      "pile-head embedment/detailing permits, use a smaller "
                      "bar, or provide a qualified headed/mechanical anchor.")
        else:
            action = ("Increase cap edge distance in X, use standard hooks/"
                      "headed bars, or use smaller diameter bars.")
        add(
            "Anchorage X",
            "Available development length is less than required.",
            action)

    if anch_y is not None and not anch_y.get("ok", True):
        if str(anch_y.get("anchorage_mode", "")).startswith("90"):
            action = ("Increase cap thickness, lower the bottom-bar layer if "
                      "pile-head embedment/detailing permits, use a smaller "
                      "bar, or provide a qualified headed/mechanical anchor.")
        else:
            action = ("Increase cap edge distance in Y, use standard hooks/"
                      "headed bars, or use smaller diameter bars.")
        add(
            "Anchorage Y",
            "Available development length is less than required.",
            action)

    if not recs and not results.get("overall_OK", True):
        add(
            "Design status",
            "The design is marked FAIL but no single dominant trigger was isolated.",
            "Review the Detail tab; check spacing, uplift, selected reinforcement, anchorage, and load direction.")

    return recs


import stm_visualization as _stm_visualization
_stm_visualization = importlib.reload(_stm_visualization)

from stm_visualization import (
    plot_layout_preview, plot_plan_view,
    plot_elevation, plot_rebar_layout, plot_3d_view,
    plot_top_rebar_layout, tie_pairs_for_3d_view,
)
from report_generator import generate_report

st.set_page_config(page_title="STM Pile Cap Designer",
                   layout="wide", page_icon="🏗️")

st.markdown(
    """
    <style>
    div[data-testid="stTabs"] div[data-baseweb="tab-list"] {
        gap: 0.35rem;
        flex-wrap: wrap;
        overflow-x: auto;
        overflow-y: visible;
        border-bottom: 2px solid #cbd5e1;
        padding: 0.25rem 0 0.45rem 0;
        margin-bottom: 0.75rem;
    }

    div[data-testid="stTabs"] div[role="tablist"] {
        display: flex;
        flex-wrap: wrap;
        gap: 0.35rem;
        overflow-x: auto;
        overflow-y: visible;
        padding-bottom: 0.45rem;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"] {
        height: 2.65rem;
        padding: 0 0.72rem;
        flex: 0 0 auto;
        border: 1px solid #d7dee8;
        border-bottom-color: #cbd5e1;
        border-radius: 8px 8px 0 0;
        background: #f8fafc;
        color: #334155;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.08);
        transition: background 120ms ease, border-color 120ms ease,
                    color 120ms ease, box-shadow 120ms ease;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"] p {
        font-size: 0.92rem;
        font-weight: 650;
        line-height: 1.1;
        white-space: nowrap;
    }

    div[data-testid="stTabs"] button[role="tab"] {
        flex: 0 0 auto;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"]:hover {
        background: #eef6ff;
        border-color: #93c5fd;
        color: #0f4c81;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.12);
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] {
        background: #0f4c81;
        border-color: #0f4c81;
        color: #ffffff;
        box-shadow: 0 3px 10px rgba(15, 76, 129, 0.25);
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"] p {
        color: #ffffff;
        font-weight: 750;
    }

    div[data-testid="stTabs"] button[data-baseweb="tab"][aria-selected="true"]::after {
        background-color: #f59e0b;
        height: 4px;
        border-radius: 999px 999px 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True)

# --------- Session-state defaults (for save/load) ---------
DEFAULTS = {
    "fc": 28.0, "fy": 420.0, "cover": 75,
    "Pu": 5000.0, "Mux": 0.0, "Muy": 0.0,
    "col_size": 500, "D": 600, "h_cap": 900,
    "col_section": "Square",
    "col_bx": 500.0, "col_by": 500.0, "col_diam": 500.0,
    "col_x": 0.0, "col_y": 0.0,
    "stm_model_type": "Point Column STM",
    "pile_section": "Circular",
    "pile_bx": 600.0, "pile_by": 600.0, "pile_diam": 600.0,
    "preset_choice": "4-Pile (Square)",
    "spacing_factor": 2.5,  # Legacy save/load key.
    "spacing_factor_x": 2.5, "spacing_factor_y": 2.5,
    "adv_spacing": False,
    "sf_x": 2.5, "sf_y": 2.5, "clear_min": 500.0,
    "e_left": 450.0, "e_right": 450.0,
    "e_top": 450.0, "e_bot": 450.0,
    "L_side": 4000, "w_trunc": 600, "e_trunc": 450.0,
    "custom_text": "1500,1500\n-1500,1500\n-1500,-1500\n1500,-1500",
    "custom_shape": "Square",
    "x_bar": "DB20", "x_n": 8,
    "y_bar": "DB20", "y_n": 8,
    "anchorage_mode": "90° Vertical Hook",
    "anchorage_bottom_z": 150.0,
    "top_bar_size": "DB20",
    "show_3d_force_labels": True,
    "wcap_uls_factor": 1.2,
}
_had_spacing_x = "spacing_factor_x" in st.session_state
_had_spacing_y = "spacing_factor_y" in st.session_state
for _k, _v in DEFAULTS.items():
    st.session_state.setdefault(_k, _v)
if not _had_spacing_x:
    st.session_state.spacing_factor_x = st.session_state.spacing_factor
if not _had_spacing_y:
    st.session_state.spacing_factor_y = st.session_state.spacing_factor

st.title("🏗️ STM Pile Cap Designer")
st.caption("Strut-and-Tie Method - ACI 318-19 / CRSI Design Handbook")

# --------- Save / Load JSON ---------
with st.sidebar:
    st.subheader("💾 Save / Load Design")
    sl1, sl2 = st.columns(2)
    state_for_save = {k: st.session_state[k] for k in DEFAULTS}
    _save_sfx = float(st.session_state.get("spacing_factor_x", 2.5))
    _save_sfy = float(st.session_state.get("spacing_factor_y", _save_sfx))
    state_for_save["spacing_factor"] = max(_save_sfx, _save_sfy)
    state_for_save["sf_x"] = _save_sfx
    state_for_save["sf_y"] = _save_sfy
    sl1.download_button(
        "💾 Save",
        data=json.dumps(state_for_save, indent=2),
        file_name="pile_cap_design.json",
        mime="application/json",
        use_container_width=True)
    up = sl2.file_uploader("📂 Open File", type="json",
                           label_visibility="visible",
                           key="json_uploader")
    if up is not None and up.file_id != st.session_state.get(
            "_last_upload_id"):
        try:
            data = json.loads(up.read().decode("utf-8"))
            for k in DEFAULTS:
                if k in data:
                    st.session_state[k] = data[k]
            if "spacing_factor" in data:
                if "spacing_factor_x" not in data:
                    st.session_state.spacing_factor_x = data["spacing_factor"]
                if "spacing_factor_y" not in data:
                    st.session_state.spacing_factor_y = data["spacing_factor"]
            st.session_state["_last_upload_id"] = up.file_id
            st.success("Loaded — refreshing...")
            st.rerun()
        except Exception as exc:
            st.error("Load failed: {}".format(exc))
    st.caption("💡 ไฟล์จะถูกบันทึกที่ Downloads folder "
               "หากต้องการเลือก folder เอง ให้เปิด "
               "\"Ask where to save\" ในการตั้งค่า Browser")

    st.divider()
    st.header("⚙️ Inputs")

    with st.expander("Materials", expanded=True):
        st.number_input("f'c (MPa)", 21.0, 80.0, step=1.0, key="fc")
        st.number_input("fy (MPa)", 280.0, 700.0, step=10.0, key="fy")
        st.number_input("Cover (mm)", 50, 200, step=5, key="cover")

    with st.expander("Column Loads", expanded=True):
        st.number_input("Pu — axial (kN)",
                        100.0, 50000.0, step=100.0, key="Pu")
        st.number_input("Mux — moment about X (kN·m)",
                        -50000.0, 50000.0, step=10.0, key="Mux",
                        help="P_i ∝ +y_i when Mux > 0")
        st.number_input("Muy — moment about Y (kN·m)",
                        -50000.0, 50000.0, step=10.0, key="Muy",
                        help="P_i ∝ +x_i when Muy > 0")
        
        st.selectbox("Column section",
                     ["Square", "Rectangular", "Circular"],
                     key="col_section")
        if st.session_state.col_section == "Square":
            st.number_input("Column side b (mm)",
                            min_value=1.0, step=50.0, key="col_bx")
            st.session_state.col_by = st.session_state.col_bx
        elif st.session_state.col_section == "Rectangular":
            cc1, cc2 = st.columns(2)
            cc1.number_input("bx (mm)", min_value=1.0,
                             step=50.0, key="col_bx")
            cc2.number_input("by (mm)", min_value=1.0,
                             step=50.0, key="col_by")
        else:
            st.number_input("Diameter D_c (mm)",
                            min_value=1.0, step=50.0, key="col_diam")
        pc1, pc2 = st.columns(2)
        pc1.number_input("Column X (mm)",
                         step=50.0, key="col_x",
                         help="Column/load point coordinate measured from layout origin.")
        pc2.number_input("Column Y (mm)",
                         step=50.0, key="col_y",
                         help="Column/load point coordinate measured from layout origin.")
        st.selectbox(
            "STM model type",
            ["Point Column STM", "Wall/Abutment STM"],
            key="stm_model_type",
            help=("Point Column STM uses the column centroid as the top node "
                  "for all struts. Wall/Abutment STM distributes the top "
                  "node along the wall length in X and uses the wall "
                  "centerline in Y at the pile-cap top face."))

    with st.expander("Pile & Cap", expanded=True):
        st.selectbox("Pile section",
                     ["Circular", "Square", "Rectangular"],
                     key="pile_section")
        if st.session_state.pile_section == "Circular":
            st.number_input("Pile diameter D (mm)",
                            min_value=1.0, step=50.0, key="pile_diam")
        elif st.session_state.pile_section == "Square":
            st.number_input("Pile side b (mm)",
                            min_value=1.0, step=50.0, key="pile_bx")
            st.session_state.pile_by = st.session_state.pile_bx
        else:
            pp1, pp2 = st.columns(2)
            pp1.number_input("Pile bx (mm)",
                             min_value=1.0, step=50.0, key="pile_bx")
            pp2.number_input("Pile by (mm)",
                             min_value=1.0, step=50.0, key="pile_by")
        st.number_input("Cap thickness (mm)", min_value=1, step=50,
                        key="h_cap")
        st.number_input(
            "W_cap ULS factor (γ)",
            min_value=1.0, max_value=2.0, step=0.05,
            key="wcap_uls_factor",
            help="W_cap(ULS) = Lx × Ly × h × 24 × γ  "
                 "ค่าแนะนำ: 1.2 (DL factor) หรือ 1.35 (Eurocode)")

    fc = st.session_state.fc
    fy = st.session_state.fy
    cover = st.session_state.cover
    Pu = st.session_state.Pu
    Mux = st.session_state.Mux
    Muy = st.session_state.Muy
    col_size = {
        "section": st.session_state.col_section,
        "bx": float(st.session_state.col_bx),
        "by": float(st.session_state.col_by),
        "diam": float(st.session_state.col_diam),
        "x": float(st.session_state.col_x),
        "y": float(st.session_state.col_y),
        "stm_model_type": st.session_state.stm_model_type,
    }
    D = {
        "section": st.session_state.pile_section,
        "bx": float(st.session_state.pile_bx),
        "by": float(st.session_state.pile_by),
        "diam": float(st.session_state.pile_diam),
    }
    h_cap = st.session_state.h_cap

    st.divider()
    st.header("📐 Pile Arrangement")

    if D["section"] == "Circular":
        _gov_max = D["diam"]
    elif D["section"] == "Square":
        _gov_max = D["bx"]
    else:
        _gov_max = max(D["bx"], D["by"])
    e_def = max(150.0, 0.75 * _gov_max)
    _preset_sfx = float(st.session_state.get(
        "spacing_factor_x", st.session_state.get("spacing_factor", 2.5)))
    _preset_sfy = float(st.session_state.get(
        "spacing_factor_y", st.session_state.get("spacing_factor", 2.5)))
    presets_init = get_preset_layouts(
        D, sf=max(_preset_sfx, _preset_sfy),
        clear_min=st.session_state.clear_min,
        sf_x=_preset_sfx, sf_y=_preset_sfy,
        e_left=e_def, e_right=e_def, e_top=e_def, e_bot=e_def)
    options = list(presets_init.keys()) + [
        "3-Pile (Truncated Triangle - Equal corners)",
        "Custom (User-defined coords)",
    ]
    if st.session_state.preset_choice not in options:
        st.session_state.preset_choice = (
            "4-Pile (Square)" if "4-Pile (Square)" in options else options[0]
        )
    st.selectbox("Preset", options, key="preset_choice")
    chosen = st.session_state.preset_choice

    is_custom = chosen.startswith("Custom")
    is_trunc = chosen.startswith("3-Pile (Truncated")

    sp1, sp2 = st.columns(2)
    sp1.number_input("X spacing factor (xD)", min_value=0.0, step=0.1,
                     key="spacing_factor_x",
                     disabled=(is_custom or is_trunc),
                     help="Center-to-center spacing along the X direction.")
    sp2.number_input("Y spacing factor (xD)", min_value=0.0, step=0.1,
                     key="spacing_factor_y",
                     disabled=(is_custom or is_trunc),
                     help="Center-to-center spacing along the Y direction.")
    st.session_state.spacing_factor = max(
        float(st.session_state.spacing_factor_x),
        float(st.session_state.spacing_factor_y))
    st.session_state.sf_x = float(st.session_state.spacing_factor_x)
    st.session_state.sf_y = float(st.session_state.spacing_factor_y)

    with st.expander("⚙️ Spacing limits / anti-overlap", expanded=False):
        st.number_input(
            "Min clear edge-to-edge (mm)",
            min_value=0.0, step=50.0, key="clear_min",
            disabled=(is_custom or is_trunc),
            help="Anti-collision: pile edges never closer than this. "
                 "Default 500 mm.")
        st.caption(
            "Auto-fix rule: sx = max(sf_x × pile_X, pile_X + clear_min), "
            "sy = max(sf_y × pile_Y, pile_Y + clear_min). "
            "For 300×1000 piles → spacing auto-expands along the long axis "
            "to prevent overlap.")

    st.markdown("**Edge distance per side (mm)**")
    c1, c2 = st.columns(2)
    c1.number_input("e_left", min_value=0.0, step=10.0, key="e_left")
    c2.number_input("e_right", min_value=0.0, step=10.0, key="e_right")
    c1.number_input("e_top", min_value=0.0, step=10.0, key="e_top")
    c2.number_input("e_bot", min_value=0.0, step=10.0, key="e_bot")
    e_left = st.session_state.e_left
    e_right = st.session_state.e_right
    e_top = st.session_state.e_top
    e_bot = st.session_state.e_bot
    spacing_factor_x = float(st.session_state.spacing_factor_x)
    spacing_factor_y = float(st.session_state.spacing_factor_y)
    spacing_factor_diag = max(spacing_factor_x, spacing_factor_y)

    coords = []
    shape_label = "Custom"
    cap_lx = cap_ly = 1500.0
    cap_cx = cap_cy = 0.0
    cap_polygon = None
    trunc_extra = ""

    if is_trunc:
        st.markdown("**Truncated Equilateral Triangle**")
        st.number_input("Side L (mm)", min_value=1, step=50, key="L_side")
        st.number_input("Truncation w (mm)", min_value=0, step=25,
                        key="w_trunc")
        st.number_input("Pile-to-edge e (mm)", min_value=0.0, step=10.0,
                        key="e_trunc")
        cfg = get_truncated_triangle_equal(
            D, float(st.session_state.L_side),
            float(st.session_state.w_trunc),
            float(st.session_state.e_trunc))
        coords = list(cfg["coords"])
        shape_label = cfg["shape"]
        cap_lx, cap_ly = cfg["lx"], cfg["ly"]
        cap_cx, cap_cy = cfg["cx"], cfg["cy"]
        cap_polygon = cfg["cap_polygon"]
        trunc_extra = (
            "d_p = R - w·√3/2 - e - D/2 = {:.0f} mm  |  "
            "Pile spacing = d_p·√3 = {:.0f} mm".format(
                cfg['d_p'], cfg['pile_spacing']))
        if cfg['d_p'] <= 0:
            st.error("Triangle too small. Increase L.")
    elif not is_custom:
        presets = get_preset_layouts(
            D, sf=spacing_factor_diag,
            clear_min=st.session_state.clear_min,
            sf_x=spacing_factor_x, sf_y=spacing_factor_y,
            e_left=e_left, e_right=e_right,
            e_top=e_top, e_bot=e_bot)
        cfg = presets[chosen]
        coords = list(cfg["coords"])
        shape_label = cfg["shape"]
        cap_lx, cap_ly = cfg["lx"], cfg["ly"]
        cap_cx, cap_cy = cfg["cx"], cfg["cy"]
    else:
        st.markdown("**Custom coordinates (mm)** (one `x, y` per line)")
        st.caption(
            "Default = 4 piles in a 3×3 m square arrangement. "
            "Edit, then click Update to apply.")
        _default_custom = "1500,1500\n-1500,1500\n-1500,-1500\n1500,-1500"
        _applied_txt = st.session_state.get("custom_text", "") or ""
        if not _applied_txt.strip():
            _applied_txt = _default_custom
            st.session_state.custom_text = _applied_txt
        if st.session_state.get("_custom_text_applied_seen") != _applied_txt:
            st.session_state.custom_text_draft = _applied_txt
            st.session_state._custom_text_applied_seen = _applied_txt
        st.text_area("Coords", height=140,
                     label_visibility="collapsed",
                     key="custom_text_draft")
        draft_txt = st.session_state.get("custom_text_draft", "")
        draft_coords = parse_custom_coords(draft_txt)
        b_apply, b_status = st.columns([1, 2])
        if b_apply.button("Update pile coordinates",
                          use_container_width=True,
                          key="apply_custom_coords"):
            if len(draft_coords) < 2:
                st.error("Please enter at least 2 valid pile coordinates.")
            else:
                st.session_state.custom_text = draft_txt
                st.session_state._custom_text_applied_seen = draft_txt
                st.session_state.pop("_stm_results", None)
                st.rerun()
        if draft_txt != _applied_txt:
            b_status.warning(
                "Draft not applied yet. Click Update to refresh the plot.")
        else:
            b_status.caption(
                "Applied coordinates: {} pile(s).".format(len(draft_coords)))
        coords = parse_custom_coords(st.session_state.custom_text)
        st.selectbox("Cap shape (visual)",
                     ["Square", "Rectangular", "Triangular"],
                     key="custom_shape")
        shape_label = st.session_state.custom_shape
        if coords:
            cap_lx, cap_ly, cap_cx, cap_cy = compute_cap_bounds_per_side(
                coords, e_left, e_right, e_top, e_bot, D)

    _col_width_x = (
        float(col_size["diam"]) if col_size["section"] == "Circular"
        else float(col_size["bx"]))
    _long_wall_ratio = _col_width_x / cap_lx if cap_lx else 0.0
    if (st.session_state.stm_model_type == "Point Column STM" and
            col_size["section"] == "Rectangular" and
            _long_wall_ratio >= 0.50):
        st.warning(
            "Long wall/abutment behavior detected: column bx is {:.0%} of "
            "cap Lx. Point Column STM may create artificial shallow struts. "
            "Consider using Wall/Abutment STM.".format(_long_wall_ratio))

    st.divider()
    st.subheader("🔩 Rebar Selection")
    bar_options = list(REBAR_DB.keys())
    cA, cB = st.columns(2)
    cA.selectbox("X-dir bar", bar_options, key="x_bar")
    cA.number_input("X-dir count", 2, 400, step=1, key="x_n")
    cB.selectbox("Y-dir bar", bar_options, key="y_bar")
    cB.number_input("Y-dir count", 2, 400, step=1, key="y_n")
    st.selectbox(
        "Anchorage mode",
        ["90° Vertical Hook", "Horizontal to edge"],
        key="anchorage_mode",
        help=("90° Vertical Hook checks the hook leg up through the cap "
              "thickness. Horizontal to edge uses the plan edge distance."))
    st.number_input(
        "Bottom bar z from cap bottom (mm)",
        min_value=0.0, step=10.0, key="anchorage_bottom_z",
        disabled=(st.session_state.anchorage_mode != "90° Vertical Hook"),
        help=("Approximate centroid level of bottom tie bars above the cap "
              "bottom. Include pile-head embedment and detailing clearance."))
    x_bar = st.session_state.x_bar
    x_n = st.session_state.x_n
    y_bar = st.session_state.y_bar
    y_n = st.session_state.y_n

    st.divider()
    calc_btn = st.button("🧮 Calculate STM",
                         type="primary", use_container_width=True)

# ===== Validation =====
ok_sp, mn_sp, viol, mn_req, pairs = validate_pile_spacing(
    coords, D, mf=2.5, clear_min=st.session_state.clear_min)
mn_clear = min((p[3] for p in pairs), default=0.0)

# ===== Layout & Status =====
left, right = st.columns([3, 2], gap="large")
with left:
    st.subheader("📍 Layout Preview (Live)")
    if not coords:
        st.warning("No piles defined.")
    else:
        W_cap_preview = _cap_area_m2(cap_polygon, cap_lx, cap_ly) \
                        * (h_cap/1000.0) * 24.0 \
                        * st.session_state.wcap_uls_factor
        pile_loads_preview = compute_pile_reactions(
            coords, Pu + W_cap_preview, Mux, Muy,
            load_point=(col_size["x"], col_size["y"]))
        st.plotly_chart(
            plot_layout_preview(
                coords, D, cap_lx, cap_ly, cap_cx, cap_cy,
                col_size, shape_label,
                cap_polygon=cap_polygon,
                edge_info=(e_left, e_right, e_top, e_bot),
                pile_loads=pile_loads_preview),
            use_container_width=True)
        if trunc_extra:
            st.info(trunc_extra)

with right:
    st.subheader("✅ Layout Status")
    if not coords:
        st.error("No piles defined.")
    else:
        c1, c2 = st.columns(2)
        c1.metric("Piles", len(coords))
        c2.metric("Shape", shape_label)
        c1.metric("Min spacing", "{:.0f} mm".format(mn_sp))
        c2.metric("Required (2.5 × D_short)",
                  "{:.0f} mm".format(mn_req),
                  help="ACI rule: 2.5 × pile_min_dim "
                       "(shorter side for Barrette piles)")
        c1.metric("Min clear", "{:.0f} mm".format(mn_clear))
        c2.metric("Required clear", "{:.0f} mm".format(st.session_state.clear_min))
        if ok_sp:
            st.success(
                "All pile-pair distances and edge clearances satisfy limits.")
        else:
            st.error("{} pile spacing/clearance violation(s).".format(len(viol)))
            for _i, _j, _d, _clear, _reason in viol[:5]:
                st.caption("P{}-P{}: {}".format(_i, _j, _reason))
        st.markdown("**Pile-Pair Distances** (d = √(Δx² + Δy²))")
        if pairs:
            pdf = pd.DataFrame([{
                "Pair": "P{}-P{}".format(i, j),
                "d (mm)": "{:.0f}".format(d),
                "Clear (mm)": "{:.0f}".format(clear),
                "c/c OK?": "✅" if ctc_ok else "❌",
                "Clear OK?": "✅" if clear_ok else "❌",
                "OK?": "✅" if ok else "❌",
            } for (i, j, d, clear, ctc_ok, clear_ok, ok) in pairs])
            st.dataframe(pdf, use_container_width=True, hide_index=True,
                         height=min(35*len(pairs)+38, 240))

st.info(
    "ℹ️ **ACI 318-19 §13.4.6.3 Note:** When the Strut-and-Tie Method (STM) "
    "is used to design a pile cap in accordance with ACI 318-19 Chapter 23, "
    "separate **beam-shear** and **two-way (punching)** shear checks are "
    "**NOT required**. The shear behavior is implicitly captured through "
    "the strength of struts and nodal zones in the STM model.")

st.divider()

# ===== Calculate =====
# ===== Calculate =====
if calc_btn:
    if not coords:
        st.error("Cannot calculate: no piles.")
    elif not ok_sp:
        st.error("Cannot calculate: spacing or pile-clearance violations.")
    else:
        # น้ำหนักตัวเอง Pile Cap (ULS): W = Area × h × γc × γ_ULS
        # Area = พื้นที่จริงของ cap (Shoelace สำหรับ polygon, Lx×Ly สำหรับสี่เหลี่ยม)
        _W_cap_nom = _cap_area_m2(cap_polygon, cap_lx, cap_ly) * (h_cap/1000.0) * 24.0
        W_cap_kN = _W_cap_nom * st.session_state.wcap_uls_factor
        _fy_x = min(REBAR_FY.get(x_bar, fy), FY_CAP_MPA)
        _fy_y = min(REBAR_FY.get(y_bar, fy), FY_CAP_MPA)
        _col_design = dict(col_size)
        _col_design["_cap_lx_mm"] = cap_lx
        _col_design["_cap_ly_mm"] = cap_ly
        _res = stm_design(coords, Pu, Mux, Muy, fc, fy, D,
                          _col_design, h_cap, cover, W_cap_kN=W_cap_kN,
                          fy_x=_fy_x, fy_y=_fy_y,
                          x_bar_size=x_bar, y_bar_size=y_bar,
                          stm_model_type=st.session_state.stm_model_type)
        if "error" in _res:
            st.error(_res["error"])
            st.session_state.pop("_stm_results", None)
        else:
            st.session_state["_stm_results"] = _res

# ===== Display (persists across reruns — dropdown-safe) =====
if "_stm_results" in st.session_state:
    results = st.session_state["_stm_results"]

    ok = results["overall_OK"]
    tag = "\u2705 DESIGN OK" if ok else "\u274c DESIGN FAILS"
    msg = ("{} - Strut DCR={:.2f}, Bearing DCR={:.2f}, "
           "Column DCR={:.2f}").format(
        tag, results["strut_DCR"],
        results["bearing_DCR"], results["column_DCR"])
    (st.success if ok else st.error)(msg)
    if results["has_uplift"]:
        st.warning("\u26a0\ufe0f Uplift detected (P_min = {:.1f} kN). "
                   "Provide tension piles or anchorage.".format(
                       results["P_min_kN"]))
    if not results.get("reaction_equilibrium_OK", True):
        st.error(
            "Pile reaction equilibrium cannot satisfy the applied moments. "
            "Residuals: Mux={:.1f} kN·m, Muy={:.1f} kN·m.".format(
                results.get("reaction_equilibrium_residual_Mux_kNm", 0.0),
                results.get("reaction_equilibrium_residual_Muy_kNm", 0.0)))
    for _warning in results.get("reaction_warnings", []):
        st.warning(_warning)
    if results.get("capacity_model_note"):
        st.info(results["capacity_model_note"])
    if (results.get("x_bar_size") and
            (results.get("x_bar_size") != x_bar or
             results.get("y_bar_size") != y_bar)):
        st.warning(
            "Bottom bar size changed after the last calculation. "
            "Click Calculate STM again to refresh As demand with the selected fy.")
    _calc_col = results.get("column_position", (0.0, 0.0))
    if (abs(_calc_col[0] - col_size["x"]) > 1e-6 or
            abs(_calc_col[1] - col_size["y"]) > 1e-6):
        st.warning(
            "Column position changed after the last calculation. "
            "Click Calculate STM again to refresh reactions, struts, and ties.")
    if results.get("stm_model_type") != st.session_state.stm_model_type:
        st.warning(
            "STM model type changed after the last calculation. "
            "Click Calculate STM again to refresh strut geometry and tie demand.")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Pu (column)",
              "{:.0f} kN".format(results["Pu_kN"]))
    m2.metric("W_cap (self-wt)",
              "{:.0f} kN".format(results["W_cap_kN"]),
              help="Lx × Ly × h × 24 kN/m³")
    m3.metric("Pu_total",
              "{:.0f} kN".format(results["Pu_total_kN"]),
              help="Pu + W_cap — used for pile reaction calculation")
    m4.metric("P_max (pile)",
              "{:.0f} kN".format(results["P_max_kN"]))
    col_x_res, col_y_res = results.get("column_position", (0.0, 0.0))
    st.caption("Column/load point = ({:.0f}, {:.0f}) mm".format(
        col_x_res, col_y_res))
    st.caption("STM model = {}. {}".format(
        results.get("stm_model_type", "Point Column STM"),
        results.get("stm_model_note", "")))
    m5, m6, m7, m8 = st.columns(4)
    m5.metric("Tie X max",
              "{:.0f} kN".format(results["F_tie_x_max_kN"]))
    m6.metric("Tie Y max",
              "{:.0f} kN".format(results["F_tie_y_max_kN"]))
    m7.metric("d_eff",
              "{:.0f} mm".format(results["d_effective_mm"]))
    m8.metric("Min strut angle",
              "{:.1f}°".format(results["min_strut_angle_deg"]),
              help="ACI requires ≥ 25°")

    _rho_lbl = "ρ_min={:.4f}·Ag".format(results.get("rho_bottom_min", 0.0018))
    x_chk = check_rebar(x_bar, int(x_n),
                        results["As_x_required_mm2"],
                        fy_mpa=results.get("fy_x_design_mpa", fy),
                        force_req_kN=results.get("F_tie_x_design_kN"),
                        rho_label=_rho_lbl)
    y_chk = check_rebar(y_bar, int(y_n),
                        results["As_y_required_mm2"],
                        fy_mpa=results.get("fy_y_design_mpa", fy),
                        force_req_kN=results.get("F_tie_y_design_kN"),
                        rho_label=_rho_lbl)

    # Auto-optimize
    opt_x, opts_x = optimize_rebar(
        results["As_x_required_mm2"], cap_lx,
        force_req_kN=results.get("F_tie_x_design_kN"),
        min_As_req=results.get("As_x_min_required_mm2"))
    opt_y, opts_y = optimize_rebar(
        results["As_y_required_mm2"], cap_ly,
        force_req_kN=results.get("F_tie_y_design_kN"),
        min_As_req=results.get("As_y_min_required_mm2"))

    # Anchorage check
    # use D/4 (inner face of CCT node) not D/2
    avail_str_x = min(e_left, e_right) + _gov_max/4.0 - cover
    avail_hook_x = avail_str_x - cover
    avail_str_y = min(e_top, e_bot) + _gov_max/4.0 - cover
    avail_hook_y = avail_str_y - cover
    anchorage_mode = st.session_state.get(
        "anchorage_mode", "90° Vertical Hook")
    bottom_bar_z = float(st.session_state.get("anchorage_bottom_z", 150.0))
    avail_vertical_hook = max(0.0, h_cap - bottom_bar_z - cover)
    use_vertical_hook = (anchorage_mode == "90° Vertical Hook")
    anch_x = _check_anchorage_with_mode(
        x_bar, fc, results.get("fy_x_design_mpa", fy),
        avail_str_x, avail_hook_x,
        anchorage_mode, avail_vertical_hook)
    anch_y = _check_anchorage_with_mode(
        y_bar, fc, results.get("fy_y_design_mpa", fy),
        avail_str_y, avail_hook_y,
        anchorage_mode, avail_vertical_hook)

    # Top-face reinforcement — always recomputed fresh (dropdown-safe)
    top_rebar = compute_top_reinforcement(
        lx_mm=cap_lx, ly_mm=cap_ly,
        h_cap_mm=h_cap, fy_mpa=fy, fc_mpa=fc,
        cover_mm=cover,
        top_bar_size=st.session_state.get("top_bar_size", "DB20"))
    top_rebar = _ensure_top_rebar_schedule(
        top_rebar, cap_lx, cap_ly, cover)

    recs = _design_recommendations(
        results, x_chk=x_chk, y_chk=y_chk,
        anch_x=anch_x, anch_y=anch_y,
        opt_x=opt_x, opt_y=opt_y, cover=cover)
    if recs and (not results["overall_OK"] or not x_chk["ok"] or
                 not y_chk["ok"] or not anch_x["ok"] or not anch_y["ok"]):
        st.warning("แนวทางปรับแบบให้ผ่านเกณฑ์")
        st.dataframe(pd.DataFrame(recs), use_container_width=True,
                     hide_index=True)

    t1, t2, t6, t3, t7, t4, t5, t8 = st.tabs([
        "📊 Plan", "📈 Elevation", "🎲 3D View",
        "🔩 Bottom Rebar", "🪟 Top Rebar",
        "⚓ Anchorage", "📋 Detail", "🧱 Pile Forces"])

    with t1:
        st.plotly_chart(
            plot_plan_view(coords, D, cap_lx, cap_ly,
                           col_size, results,
                           cap_cx, cap_cy, cap_polygon),
            use_container_width=True)
        st.markdown("**Pile Reactions** (rigid-cap formula)")
        _formula_box(
            "P_i = Pu_total/n + Mux,c*y'_i / sum(y'_j^2) "
            "+ Muy,c*x'_i / sum(x'_j^2)")
        rdf = pd.DataFrame([{
            "Pile": "P{}".format(i+1),
            "X (mm)": "{:.0f}".format(c[0]),
            "Y (mm)": "{:.0f}".format(c[1]),
            "x' (mm)": "{:.0f}".format(
                results.get("reaction_coords", [(0, 0)]*len(coords))[i][0]),
            "y' (mm)": "{:.0f}".format(
                results.get("reaction_coords", [(0, 0)]*len(coords))[i][1]),
            "P_i (kN)": "{:.1f}".format(P),
        } for i, (c, P) in enumerate(
            zip(coords, results["pile_loads_kN"]))])
        st.dataframe(rdf, use_container_width=True, hide_index=True)

    with t2:
        st.plotly_chart(
            plot_elevation(coords, h_cap, D, col_size, results),
            use_container_width=True)
    
    with t6:
        st.plotly_chart(
            plot_3d_view(coords, D, cap_lx, cap_ly,
                         cap_cx, cap_cy, col_size, h_cap,
                         cap_polygon, results,
                         show_force_labels=False,
                         show_member_labels=True),
            use_container_width=True)
        st.caption(
            "🖱️ **Drag** to rotate | **Scroll** to zoom | "
            "**Shift+drag** to pan | **Double-click** to reset view | "
            "Hover on struts/ties to inspect force values.")
        st.markdown(
            "**Color legend (Struts):** "
            "🟢 DCR < 60% | 🟠 60–85% | 🔴 > 85%  &nbsp;&nbsp; "
            "**Member labels:** `S#` = strut, `T#` = tie path, `P#` = pile. "
            "Numerical design forces are listed in the tables below.")
        force_summary_rows = _design_force_summary_rows(results)
        if force_summary_rows:
            st.markdown("### Design Force Summary")
            st.dataframe(
                pd.DataFrame(force_summary_rows),
                use_container_width=True, hide_index=True)
            st.caption(
                "Strut force is the maximum compression demand used for the "
                "strut DCR check. Tie force is the STM demand used for bottom "
                "reinforcement before comparing with minimum reinforcement.")
        strut_force_rows = _strut_force_rows(results)
        if strut_force_rows:
            st.markdown("### Strut Forces by Member")
            st.dataframe(
                pd.DataFrame(strut_force_rows),
                use_container_width=True, hide_index=True)
        tie_member_rows = _tie_member_rows(coords, results)
        if tie_member_rows:
            st.markdown("### Tie Forces by Member")
            st.dataframe(
                pd.DataFrame(tie_member_rows),
                use_container_width=True, hide_index=True)
            st.caption(
                "Tie member forces are estimated for the displayed T# paths "
                "from cut equilibrium of local strut components. Use the "
                "Design Force Summary for governing bottom reinforcement.")
    
    with t3:
        st.markdown("### Required Reinforcement")
        st.caption(
            "Bottom-face steel is checked per direction against both STM tie "
            "demand and minimum flexural reinforcement: "
            "As_req = max(As_STM, ρ_min·Ag)  "
            "โดย ρ_min = max(0.0018×420/fy, 0.0014)  [ACI §9.6.1.2]")
        if results.get("is_3pile_resultant"):
            tie_x_display = tie_y_display = results["F_tie_res_kN"]
            note = " (Resultant)"
        else:
            tie_x_display = results["F_tie_x_max_kN"]
            tie_y_display = results["F_tie_y_max_kN"]
            note = ""
        x_spacing = _rebar_spacing_text(
            x_n, cap_ly, cap_lx, cap_ly, cover)
        y_spacing = _rebar_spacing_text(
            y_n, cap_lx, cap_lx, cap_ly, cover)
        req_df = pd.DataFrame([
            {"Direction": "X"+note,
             "Tie (kN)":
                 "{:.1f}".format(tie_x_display),
             "As STM (mm²)":
                 "{:.0f}".format(results.get("As_x_stm_required_mm2", 0.0)),
             "As min ρ_min·Ag (mm²)":
                 "{:.0f}".format(results.get("As_x_min_required_mm2", 0.0)),
             "As req = max (mm²)":
                 "{:.0f}".format(results["As_x_required_mm2"]),
             "Governs":
                 results.get("As_x_governs", "—"),
             "fy used (MPa)":
                 "{:.0f}".format(results.get("fy_x_design_mpa", fy)),
             "Selected": "{}-{}".format(int(x_n), x_bar),
             "Spacing s (mm)":
                 x_spacing,
             "As prov (mm²)":
                 "{:.0f}".format(x_chk["As_provided"]),
             "Min OK":
                 "✅" if x_chk["As_provided"] >= results.get("As_x_min_required_mm2", 0.0) else "❌",
             "STM OK":
                 "✅" if x_chk.get("force_ok", True) else "❌",
             "Ratio": "{:.2f}".format(x_chk["ratio"]),
             "Status": "✅ OK" if x_chk["ok"] else "❌ FAIL"},
            {"Direction": "Y"+note,
             "Tie (kN)":
                 "{:.1f}".format(tie_y_display),
             "As STM (mm²)":
                 "{:.0f}".format(results.get("As_y_stm_required_mm2", 0.0)),
             "As min ρ_min·Ag (mm²)":
                 "{:.0f}".format(results.get("As_y_min_required_mm2", 0.0)),
             "As req = max (mm²)":
                 "{:.0f}".format(results["As_y_required_mm2"]),
             "Governs":
                 results.get("As_y_governs", "—"),
             "fy used (MPa)":
                 "{:.0f}".format(results.get("fy_y_design_mpa", fy)),
             "Selected": "{}-{}".format(int(y_n), y_bar),
             "Spacing s (mm)":
                 y_spacing,
             "As prov (mm²)":
                 "{:.0f}".format(y_chk["As_provided"]),
             "Min OK":
                 "✅" if y_chk["As_provided"] >= results.get("As_y_min_required_mm2", 0.0) else "❌",
             "STM OK":
                 "✅" if y_chk.get("force_ok", True) else "❌",
             "Ratio": "{:.2f}".format(y_chk["ratio"]),
             "Status": "✅ OK" if y_chk["ok"] else "❌ FAIL"},
        ])
        st.dataframe(req_df, use_container_width=True,
                     hide_index=True)

        # ── ρ_min note ──────────────────────────────────────────────
        _rho_min_val = results.get("rho_bottom_min", 0.0018)
        _fy_min_val  = min(results.get("fy_x_design_mpa", fy),
                           results.get("fy_y_design_mpa", fy))
        st.caption(
            "📐 **เหล็กล่าง:** ρ_min = max(0.0018 × 420 / fy, 0.0014) "
            "= max(0.0018 × 420 / {:.0f}, 0.0014) = **{:.5f}** "
            " [ACI 318-19 §9.6.1.2]  "
            "→  As_min = ρ_min × Ag = {:.5f} × Ag".format(
                _fy_min_val, _rho_min_val, _rho_min_val))

        if results.get("is_3pile_resultant"):
            st.info("ฐาน 3 เข็ม: แรงออกแบบใช้ F_res = √(Ftx²+Fty²) = {:.1f} kN; As แต่ละทิศคำนวณด้วย fy ของเหล็กที่เลือกในทิศนั้น".format(results["F_tie_res_kN"]))

        st.markdown("### 🤖 Auto-Optimized Rebar (Min Weight)")
        col_o1, col_o2, col_o3 = st.columns([2, 2, 1])
        col_o1.success(
            "**X-dir optimal:** {}-{} | "
            "As={:.0f} mm² | Wt={:.2f} kg".format(
                opt_x["n_bars"], opt_x["bar_size"],
                opt_x["As_provided"], opt_x["weight_kg"]))
        col_o2.success(
            "**Y-dir optimal:** {}-{} | "
            "As={:.0f} mm² | Wt={:.2f} kg".format(
                opt_y["n_bars"], opt_y["bar_size"],
                opt_y["As_provided"], opt_y["weight_kg"]))
        if col_o3.button("Back To Rebar Selection", use_container_width=True):
            st.session_state.x_bar = opt_x["bar_size"]
            st.session_state.x_n = int(opt_x["n_bars"])
            st.session_state.y_bar = opt_y["bar_size"]
            st.session_state.y_n = int(opt_y["n_bars"])
            st.rerun()

        with st.expander("All optimization candidates"):
            cox, coy = st.columns(2)
            cox.markdown("**X-direction**")
            cox.dataframe(pd.DataFrame([{
                "Bar": o["bar_size"], "n": o["n_bars"],
                "As req (mm²)": "{:.0f}".format(o["As_required"]),
                "As (mm²)": "{:.0f}".format(o["As_provided"]),
                "Governs": o.get("governs", "—"),
                "Wt (kg)": "{:.2f}".format(o["weight_kg"]),
                "OK": "✅" if o["ok"] else "❌",
            } for o in opts_x]), hide_index=True,
                use_container_width=True)
            coy.markdown("**Y-direction**")
            coy.dataframe(pd.DataFrame([{
                "Bar": o["bar_size"], "n": o["n_bars"],
                "As req (mm²)": "{:.0f}".format(o["As_required"]),
                "As (mm²)": "{:.0f}".format(o["As_provided"]),
                "Governs": o.get("governs", "—"),
                "Wt (kg)": "{:.2f}".format(o["weight_kg"]),
                "OK": "✅" if o["ok"] else "❌",
            } for o in opts_y]), hide_index=True,
                use_container_width=True)

        st.markdown("### Reinforcement Layout (Plan)")
        st.plotly_chart(
            plot_rebar_layout(coords, D, cap_lx, cap_ly,
                cap_cx, cap_cy, col_size, cap_polygon, results,
                x_bar, int(x_n), y_bar, int(y_n), x_chk, y_chk),
            use_container_width=True)

    with t4:
        st.markdown("### Anchorage / Development Length")
        st.markdown(
            "Per **ACI 318-19 §23.8.3** tie reinforcement must "
            "develop fy at the point where the centroid of the tie "
            "crosses the extended nodal zone (CCT node above pile).")
        st.caption(
            "Anchorage mode: {} | Bottom bar z = {:.0f} mm | "
            "Vertical hook available = {:.0f} mm".format(
                anchorage_mode, bottom_bar_z, avail_vertical_hook))
        _formula_box(
            "ld  ≈ (fy·ψs / 1.1·λ·√f'c · (cb+Ktr)/db) · db   "
            "(ACI 25.4.2.3)\n"
            "ldh ≈ (0.24·fy·db) / (λ·√f'c)                  "
            "(ACI 25.4.3.1a, SI)")

        anch_df = pd.DataFrame([
            {"Direction": "X", "Bar": anch_x["bar_size"],
             "ld req (mm)":
                 "{:.0f}".format(anch_x["ld_required_mm"]),
             "ldh req (mm)":
                 "{:.0f}".format(anch_x["ldh_required_mm"]),
             "Avail. straight":
                 "{:.0f}".format(anch_x["available_straight_mm"]),
             "Avail. hook":
                 "{:.0f}".format(anch_x["available_hook_mm"]),
             "Mode": anch_x.get("anchorage_mode", "—"),
             "Recommended": anch_x["recommended"],
             "Status": "✅" if anch_x["ok"] else "❌"},
            {"Direction": "Y", "Bar": anch_y["bar_size"],
             "ld req (mm)":
                 "{:.0f}".format(anch_y["ld_required_mm"]),
             "ldh req (mm)":
                 "{:.0f}".format(anch_y["ldh_required_mm"]),
             "Avail. straight":
                 "{:.0f}".format(anch_y["available_straight_mm"]),
             "Avail. hook":
                 "{:.0f}".format(anch_y["available_hook_mm"]),
             "Mode": anch_y.get("anchorage_mode", "—"),
             "Recommended": anch_y["recommended"],
             "Status": "✅" if anch_y["ok"] else "❌"},
        ])
        st.dataframe(anch_df, use_container_width=True,
                     hide_index=True)

        if not (anch_x["ok"] and anch_y["ok"]):
            if use_vertical_hook:
                st.warning(
                    "⚠️ Vertical hook anchorage insufficient. Consider: "
                    "(a) increasing cap thickness, "
                    "(b) reducing bottom bar z if detailing permits, "
                    "(c) using smaller bar diameter, or "
                    "(d) headed/mechanical anchorage.")
            else:
                st.warning(
                    "⚠️ Anchorage insufficient. Consider: "
                    "(a) larger cap dimensions, "
                    "(b) using 90°/180° standard hooks, "
                    "(c) headed bars (ACI 25.4.4), "
                    "or (d) smaller bar diameter.")

        with st.expander("Assumptions used"):
            st.markdown(
                "- λ = 1.0 (normal-weight concrete)\n"
                "- ψt=1.0 (bottom bars), ψe=1.0 (uncoated), "
                "ψg=1.0 (Gr 420), ψr=ψo=ψc=1.0\n"
                "- (cb+Ktr)/db = 1.5 (conservative typical value)\n"
                "- Available straight = min edge dist + D/4 − cover  # PATCHED: inner face of CCT node\n"
                "- Horizontal edge hook = available straight − cover\n"
                "- 90° vertical hook = h_cap − bottom bar z − top cover")

    with t7:
        st.markdown("## 🪟 Top-Face Minimum Reinforcement")
        st.markdown(
            "เหล็กผิวบนของ pile cap ไม่ได้รับแรงดึงจาก STM โดยตรง "
            "และ pile cap ที่หนามากมักไม่เกิด top-face tension จากน้ำหนักแนวดิ่ง "
            "ดังนั้นจึงไม่ควรใช้ ρ_min เต็มค่า (จะได้เหล็กมากเกินจำเป็น) "
            "→ ใช้ **ρ_top = ρ_min / 2** ในแต่ละทิศทาง  "
            "โดย ρ_min = max(0.0018 × 420/fy, 0.0014) [ACI §9.6.1.2]")

        # ── Bar selector ─────────────────────────────────────
        st.markdown("### เลือกขนาดเหล็กผิวบน")
        _bar_options = list(REBAR_DB.keys())  # DB12…DB32
        _cur_bar = st.session_state.get("top_bar_size", "DB20")
        _sel_col, _info_col = st.columns([2, 3])
        with _sel_col:
            _chosen_bar = st.selectbox(
                "Top bar size",
                options=_bar_options,
                index=_bar_options.index(_cur_bar),
                key="top_bar_size",
                help="เหล็ก ≤ DB28 → fy = 390 MPa | DB32 → fy = 490 MPa")
        with _info_col:
            _fy_chosen = REBAR_FY[_chosen_bar]
            _fy_capped = min(_fy_chosen, FY_CAP_MPA)
            if _fy_chosen > 420:
                st.warning(
                    "**{} : fy = {:.0f} MPa > 420 MPa**  \n"
                    "ACI §20.2.2.4 → cap ที่ {:.0f} MPa  \n"
                    "ACI §24.3.2   → spacing limit เพิ่มเติม".format(
                        _chosen_bar, _fy_chosen, FY_CAP_MPA))
            else:
                _fy_d_preview = min(_fy_chosen, FY_CAP_MPA)
                _rho_min_preview = max(0.0018 * 420.0 / _fy_d_preview, 0.0014)
                _rho_top_preview = _rho_min_preview / 2.0
                st.info(
                    "**{} : fy = {:.0f} MPa ≤ 420 MPa**  \n"
                    "ρ_min = max(0.0018×420/{:.0f}, 0.0014) = {:.5f}  \n"
                    "ρ_top = ρ_min / 2 = {:.5f}  |  ไม่มี spacing penalty".format(
                        _chosen_bar, _fy_chosen,
                        _fy_d_preview, _rho_min_preview, _rho_top_preview))

        # ── Recompute with selected bar ───────────────────────
        tr = compute_top_reinforcement(
            lx_mm=cap_lx, ly_mm=cap_ly,
            h_cap_mm=h_cap, fy_mpa=fy, fc_mpa=fc,
            cover_mm=cover, top_bar_size=_chosen_bar)
        tr = _ensure_top_rebar_schedule(tr, cap_lx, cap_ly, cover)
        top_rebar = tr

        # ── Code Reference Expander ──────────────────────────
        with st.expander("📖 Code Basis (ACI 318-19) — คลิกเพื่อดูรายละเอียด"):
            st.markdown("""
**Check A — Minimum Reinforcement  §9.6.1.2**

Ag_x = ly × h_cap  |  Ag_y = lx × h_cap

ρ_min = max(0.0018 × 420 / fy, 0.0014)   [ACI §9.6.1.2, fy-dependent]

As_min,bottom = ρ_min × Ag

**Top-face (pile cap หนา — ไม่เกิด top tension):**

ρ_top = ρ_min / 2   →   As_top = ρ_top × Ag

**Spacing Limits**

§24.4.3.3 : s ≤ min(3h, 450 mm)
§24.3.2   : s ≤ min(380×(280/fs), 300×(280/fs))  where fs = (2/3)fy_d
    (only becomes active when fy_d > 420 MPa)
§20.2.2.4 : fy_d used in design capped at 550 MPa

            """)

        # ── Numeric Results ──────────────────────────────────
        st.markdown("### ผลการคำนวณ  (fy_d = {:.0f} MPa — {})".format(
            tr["fy_design_mpa"], tr["fy_note"]))

        _top_bar = tr.get("top_bar_size", _chosen_bar)
        _top_x_detail = "{}-{} @ {:.0f} mm (As={:.0f})".format(
            int(tr.get("top_x_n_bars", 2)), _top_bar,
            tr.get("top_x_spacing_mm", 0.0),
            tr.get("top_x_As_provided_mm2", 0.0))
        _top_y_detail = "{}-{} @ {:.0f} mm (As={:.0f})".format(
            int(tr.get("top_y_n_bars", 2)), _top_bar,
            tr.get("top_y_spacing_mm", 0.0),
            tr.get("top_y_As_provided_mm2", 0.0))

        res_df = pd.DataFrame([
            {"Check": "Ag gross strip",
             "ρ used": "—",
             "As_X req (mm²)": "{:.0f}".format(tr["Ag_x_mm2"]),
             "As_Y req (mm²)": "{:.0f}".format(tr["Ag_y_mm2"]),
             "ใช้เป็น As_top": "ฐานคำนวณ"},
            {"Check": "ρ_min × Ag  (full bottom min)",
             "ρ used": "{:.5f}".format(tr["rho_full_min"]),
             "As_X req (mm²)": "{:.0f}".format(tr["As_full_min_x_mm2"]),
             "As_Y req (mm²)": "{:.0f}".format(tr["As_full_min_y_mm2"]),
             "ใช้เป็น As_top": "หาร 2"},
            {"Check": "ρ_top = ρ_min/2  (top face)",
             "ρ used": "{:.5f}".format(tr["rho_top"]),
             "As_X req (mm²)": "{:.0f}".format(tr["As_top_x_mm2"]),
             "As_Y req (mm²)": "{:.0f}".format(tr["As_top_y_mm2"]),
             "ใช้เป็น As_top": "✅ Yes"},
            {"Check": "Recommended top bars",
             "ρ used": "—",
             "As_X req (mm²)": _top_x_detail,
             "As_Y req (mm²)": _top_y_detail,
             "ใช้เป็น As_top": "Detailing"},
        ])
        st.dataframe(res_df, use_container_width=True, hide_index=True)

        # ── ρ_min note ───────────────────────────────────────────────
        _fy_top_d = tr.get("fy_design_mpa", 390.0)
        st.caption(
            "📐 **เหล็กบน:** ρ_min = max(0.0018 × 420 / fy, 0.0014) "
            "= max(0.0018 × 420 / {:.0f}, 0.0014) = **{:.5f}**"
            "  →  ρ_top = ρ_min / 2 = **{:.5f}**"
            "  [ACI 318-19 §9.6.1.2]  "
            "*(pile cap หนา ไม่เกิด top tension → ใช้ครึ่งค่า)*".format(
                _fy_top_d,
                tr["rho_full_min"],
                tr["rho_top"]))

        c1, c2 = st.columns(2)
        c1.success(
            "**X-dir  As_top = {:.0f} mm²**  \n"
            "Governs: {}".format(
                tr["As_top_x_mm2"], tr["governs_x"]))
        c2.success(
            "**Y-dir  As_top = {:.0f} mm²**  \n"
            "Governs: {}".format(
                tr["As_top_y_mm2"], tr["governs_y"]))
        st.caption(tr["top_design_note"])

        # ── Spacing Check ────────────────────────────────────
        st.markdown("### ตรวจสอบระยะห่างสูงสุด")
        _s_ts   = tr["s_ts_max_mm"]
        _s_cr   = tr["s_crack_mm"]
        _s_gov  = tr["s_max_top_mm"]
        _fs     = tr["fs_service_mpa"]
        spac_df = pd.DataFrame([
            {"เกณฑ์": "§24.4.3.3  min(3h, 450)",
             "s_max (mm)": "{:.0f}".format(_s_ts),
             "Active": "✅ เสมอ"},
            {"เกณฑ์": "§24.3.2  crack-width  (fs={:.0f} MPa)".format(_fs),
             "s_max (mm)": "{:.0f}".format(_s_cr),
             "Active": "⚠️ fy > 420" if tr["fy_design_mpa"] > 420
                       else "— (fy ≤ 420)"},
        ])
        st.dataframe(spac_df, use_container_width=True, hide_index=True)
        if tr["fy_design_mpa"] > 420:
            st.warning(
                "**s_max governing = {:.0f} mm** "
                "(§24.3.2 controls เนื่องจาก fy = {:.0f} MPa)".format(
                    _s_gov, tr["fy_design_mpa"]))
        else:
            st.info(
                "**s_max governing = {:.0f} mm** "
                "(§24.4.3.3 controls)".format(_s_gov))

        # ── Bar Suggestions ──────────────────────────────────
        st.markdown("### แนะนำขนาดเหล็กผิวบน  (Bar = {})".format(
            _chosen_bar))
        _top_bars = ("DB12", "DB16", "DB20", "DB25", "DB28", "DB32")
        _top_sx = suggest_rebar(tr["As_top_x_mm2"],
                                preferred=_top_bars)
        _top_sy = suggest_rebar(tr["As_top_y_mm2"],
                                preferred=_top_bars)

        def _spac_check(n, width, s_max):
            """Estimate actual spacing and flag if over limit."""
            if n <= 1:
                return "—"
            s_act = (width - 2*cover) / (n - 1)
            ok = s_act <= s_max
            return "{:.0f} mm {}".format(s_act, "✅" if ok else "❌ > {:.0f}".format(s_max))

        sug_df = pd.DataFrame([
            {"ทิศทาง": "X", "ขนาด": sz,
             "fy (MPa)": "{:.0f}".format(REBAR_FY.get(sz, fy)),
             "จำนวน (เส้น)": n,
             "As prov (mm²)": "{:.0f}".format(a),
             "As req (mm²)": "{:.0f}".format(tr["As_top_x_mm2"]),
             "Ratio": "{:.2f}".format(
                 a / tr["As_top_x_mm2"] if tr["As_top_x_mm2"] else 0),
             "Spacing (est.)": _spac_check(n, cap_ly, _s_gov),
             "Status": "✅" if a >= tr["As_top_x_mm2"] else "❌"}
            for sz, n, a in _top_sx
        ] + [
            {"ทิศทาง": "Y", "ขนาด": sz,
             "fy (MPa)": "{:.0f}".format(REBAR_FY.get(sz, fy)),
             "จำนวน (เส้น)": n,
             "As prov (mm²)": "{:.0f}".format(a),
             "As req (mm²)": "{:.0f}".format(tr["As_top_y_mm2"]),
             "Ratio": "{:.2f}".format(
                 a / tr["As_top_y_mm2"] if tr["As_top_y_mm2"] else 0),
             "Spacing (est.)": _spac_check(n, cap_lx, _s_gov),
             "Status": "✅" if a >= tr["As_top_y_mm2"] else "❌"}
            for sz, n, a in _top_sy
        ])
        st.dataframe(sug_df, use_container_width=True,
                     hide_index=True)

        # ── Plan View Diagram ────────────────────────────────
        st.markdown("### แผนผังเหล็กผิวบน (Plan View)")
        st.plotly_chart(
            plot_top_rebar_layout(
                coords, D, cap_lx, cap_ly,
                cap_cx, cap_cy, col_size, cap_polygon,
                tr, cover_mm=cover),
            use_container_width=True)

        # ── Placement Guide ──────────────────────────────────
        st.markdown("### ตำแหน่งและรูปแบบการวาง")
        st.markdown("""
| ตำแหน่ง | รายละเอียด | อ้างอิง ACI |
|---|---|---|
| **ผิวบน แนว X** | วางแนวนอน กระจายตลอดความกว้าง ly | §24.4.3.2 |
| **ผิวบน แนว Y** | วางแนวขวาง กระจายตลอดความกว้าง lx | §24.4.3.2 |
| **ระยะ cover ผิวบน** | ≥ 50 mm (exposed) ตามสภาพแวดล้อม | §20.6.1.3 |
| **ระยะห่างเหล็ก** | ≤ s_max governing (ดูตาราง spacing ด้านบน) | §24.4.3.3, §24.3.2 |
| **ต่อทับ** | ≥ 1.3 × ld (Class B splice) | §25.5.2 |
| **fy ใช้ออกแบบ** | ≤DB28 → 390 MPa, DB32 → 490 MPa, cap 550 MPa | §20.2.2.4 |
""")
        st.info(
            "💡 **หมายเหตุ:** เหล็กล่างยังต้องตรวจเต็มค่า ρ_min·Ag แยกจาก STM tie demand "
            "โดย ρ_min = max(0.0018×420/fy, 0.0014) [ACI §9.6.1.2] "
            "ส่วนเหล็กบนในหน้านี้ใช้ครึ่งหนึ่งของ minimum gross-area ตามที่กำหนด")

    with t5:
        st.markdown("### Strut Forces")
        rows = []
        for i, s in enumerate(results["struts"]):
            node_x, node_y = s.get("column_coord", (0.0, 0.0))
            rows.append({
                "Strut": "S{}".format(i+1),
                "X (mm)": round(s["coord"][0],1), "Y (mm)": round(s["coord"][1],1),
                "Node X (mm)": round(node_x, 1),
                "Node Y (mm)": round(node_y, 1),
                "dx_col (mm)": round(s.get("dx_from_col", s["coord"][0]), 1),
                "dy_col (mm)": round(s.get("dy_from_col", s["coord"][1]), 1),
                "P_i (kN)": round(s["P_i_kN"], 1),
                "L (mm)": round(s["L_strut"], 0),
                "θ (°)": round(s["theta_deg"], 1),
                "F_strut (kN)": round(s["F_strut_kN"], 1),
            })
        st.dataframe(pd.DataFrame(rows),
                     use_container_width=True, hide_index=True)

        st.markdown("### Tie Design Forces (for reinforcement)")
        tie_rows = [
            {"Direction": "X",
             "Tie force (kN)": "{:.1f}".format(results["F_tie_x_max_kN"]),
             "Formula": "Σ Pi·|dx_col|/d (controlling side)"},
            {"Direction": "Y",
             "Tie force (kN)": "{:.1f}".format(results["F_tie_y_max_kN"]),
             "Formula": "Σ Pi·|dy_col|/d (controlling side)"},
        ]
        if results.get("is_3pile_resultant"):
            tie_rows.append({
                "Direction": "Resultant (3-pile)",
                "Tie force (kN)": "{:.1f}".format(results["F_tie_res_kN"]),
                "Formula": "√(Ftx²+Fty²) — used for As_x = As_y"
            })
        tie_df = pd.DataFrame(tie_rows)
        st.dataframe(tie_df, use_container_width=True, hide_index=True)
        if results.get("is_3pile_resultant"):
            st.info("ฐาน 3 เข็ม: ใช้แรงลัพธ์ F_res เพื่อลดผลของการหมุนพิกัด แล้วคำนวณ As แยกตาม fy ของเหล็กแต่ละทิศ")
        st.caption("หมายเหตุ: ตาราง Strut แสดงแรงอัดในแต่ละ strut เท่านั้น ส่วนแรงดึงที่ใช้ออกแบบเหล็กคือค่า Tie ด้านบน")

        st.markdown("### Capacity Checks")
        _bn = results.get("bn_pile", 0.60)
        _node_lbl = "CTT βn={:.2f}".format(_bn) if results["n_piles"] >= 4 else "CCT βn={:.2f}".format(_bn)
        ck = pd.DataFrame([
            {"Check": "Strut compression",
             "Capacity (kN)":
                 "{:.0f}".format(results["phi_Fns_kN"]),
             "Demand (kN)":
                 "{:.0f}".format(results["F_strut_max_kN"]),
             "DCR": "{:.2f}".format(results["strut_DCR"]),
             "Status": "✅" if results["strut_DCR"] <= 1
                       else "❌"},
            {"Check": "Pile bearing ({})".format(_node_lbl),
             "Capacity (kN)":
                 "{:.0f}".format(results["phi_Pn_bearing_kN"]),
             "Demand (kN)":
                 "{:.0f}".format(results["P_max_kN"]),
             "DCR": "{:.2f}".format(results["bearing_DCR"]),
             "Status": "✅" if results["bearing_DCR"] <= 1
                       else "❌"},
            {"Check": "Column bearing (CCC)",
             "Capacity (kN)":
                 "{:.0f}".format(results["phi_Pn_column_kN"]),
             "Demand (kN)":
                 "{:.0f}".format(results["Pu_kN"]),
             "DCR": "{:.2f}".format(results["column_DCR"]),
             "Status": "✅" if results["column_DCR"] <= 1
                       else "❌"},
            {"Check": "Strut angle ≥25° ({:.1f}°)".format(
                results["min_strut_angle_deg"]),
             "Capacity (kN)": "-", "Demand (kN)": "-", "DCR": "-",
             "Status": "✅" if results["angle_OK"] else "❌"},
        ])
        extra_checks = [
            {"Check": "Uplift / tension pile",
             "Capacity (kN)": "-", "Demand (kN)": "{:.0f}".format(results["P_min_kN"]),
             "DCR": "-", "Status": "✅" if not results["has_uplift"] else "❌"},
            {"Check": "Reaction equilibrium",
             "Capacity (kN)": "-", "Demand (kN)": "-", "DCR": "-",
             "Status": "✅" if results.get("reaction_equilibrium_OK", True) else "❌"},
        ]
        ck = pd.concat([ck, pd.DataFrame(extra_checks)], ignore_index=True)
        st.dataframe(ck, use_container_width=True, hide_index=True)

    with t8:
        st.markdown("### Pile Head Forces Below Pile Cap")
        st.caption(
            "Forces are derived from the selected STM strut geometry. "
            "Axial P_i is the rigid-cap pile reaction; H_x and H_y are the "
            "signed horizontal components of each pile strut at the pile head.")
        _formula_box(
            "H_x,i = P_i(compression) x dx_i / d_eff\n"
            "H_y,i = P_i(compression) x dy_i / d_eff\n"
            "H_res,i = sqrt(H_x,i^2 + H_y,i^2)\n"
            "F_strut,i = P_i(compression) x L_strut,i / d_eff\n\n"
            "Sign convention: +H_x / +H_y follows the direction from the "
            "selected top compression node toward the pile head.")

        pile_force_rows, pile_force_summary = _pile_head_force_rows(results)
        if pile_force_summary:
            st.markdown("### Critical Pile Summary")
            st.dataframe(
                pd.DataFrame(pile_force_summary),
                use_container_width=True, hide_index=True)

        if pile_force_rows:
            st.markdown("### Forces by Pile")
            st.dataframe(
                pd.DataFrame(pile_force_rows),
                use_container_width=True, hide_index=True)
            st.info(
                "Use `Axial P_i` together with `H resultant` or directional "
                "`H_x/H_y` for preliminary pile-head reinforcement design. "
                "Pile-head moment is not calculated here because it depends "
                "on pile fixity, embedment, connection detailing, and soil/"
                "substructure stiffness assumptions.")

    # Export Report
    st.divider()
    st.subheader("📄 Export Report")
    inputs_dict = {
        "fc": fc, "fy": fy, "cover": cover,
        "Pu": Pu, "Mux": Mux, "Muy": Muy,
        "W_cap_kN": results["W_cap_kN"],
        "W_cap_nom_kN": _cap_area_m2(cap_polygon, cap_lx, cap_ly) * (h_cap/1000.0) * 24.0,
        "wcap_uls_factor": st.session_state.wcap_uls_factor,
        "Pu_total_kN": results["Pu_total_kN"],
        "anchorage_mode": anchorage_mode,
        "anchorage_bottom_z": bottom_bar_z,
        "anchorage_vertical_hook_avail": avail_vertical_hook,
        "stm_model_type": results.get("stm_model_type", st.session_state.stm_model_type),
        "D": D, "h_cap": h_cap, "col_size": col_size,
        "coords": coords,
        "cap_lx": cap_lx, "cap_ly": cap_ly,
        "cap_cx": cap_cx, "cap_cy": cap_cy,
        "cap_polygon": cap_polygon,
        "shape_label": shape_label,
        "spacing_factor_x": spacing_factor_x,
        "spacing_factor_y": spacing_factor_y,
        "clear_min": st.session_state.clear_min,
    }
    try:
        docx_buf = generate_report(
            inputs_dict, results, x_chk, y_chk, pairs,
            anch_x=anch_x, anch_y=anch_y,
            opt_x=opt_x, opt_y=opt_y,
            top_rebar=top_rebar)
        st.download_button(
            "⬇️ Download Word Report (.docx)",
            data=docx_buf,
            file_name="pile_cap_design_report.docx",
            mime=("application/vnd.openxmlformats-officedocument."
                  "wordprocessingml.document"),
            use_container_width=True, type="secondary")
    except Exception as exc:
        st.error("Report generation failed: {}".format(exc))
else:
    st.info("Set inputs in sidebar and click **Calculate STM**. "
            "Layout preview & pile reactions update live.")
