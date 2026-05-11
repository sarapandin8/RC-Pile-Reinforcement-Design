import math
from dataclasses import dataclass

import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Pile Reinforcement Designer",
    page_icon="P",
    layout="wide",
)


@dataclass
class SectionGeometry:
    shape: str
    area_mm2: float
    gross_width_mm: float | None = None
    gross_depth_mm: float | None = None
    gross_diameter_mm: float | None = None


def rebar_area(bar_diameter_mm: float) -> float:
    return math.pi * bar_diameter_mm**2 / 4.0


def section_geometry(shape: str, width_mm: float | None, depth_mm: float | None, diameter_mm: float | None) -> SectionGeometry:
    if shape == "Square":
        area = width_mm * width_mm
        return SectionGeometry(shape=shape, area_mm2=area, gross_width_mm=width_mm, gross_depth_mm=width_mm)
    if shape == "Rectangle":
        area = width_mm * depth_mm
        return SectionGeometry(shape=shape, area_mm2=area, gross_width_mm=width_mm, gross_depth_mm=depth_mm)

    area = math.pi * diameter_mm**2 / 4.0
    return SectionGeometry(shape=shape, area_mm2=area, gross_diameter_mm=diameter_mm)


def phi_factor(column_type: str) -> float:
    return 0.65 if column_type == "Tied" else 0.75


def effective_cover(cover_mm: float, tie_diameter_mm: float, main_bar_diameter_mm: float) -> float:
    return cover_mm + tie_diameter_mm + main_bar_diameter_mm / 2.0


def usable_core_dimension(dimension_mm: float, clear_cover_mm: float, tie_diameter_mm: float, main_bar_diameter_mm: float) -> float:
    return dimension_mm - 2.0 * effective_cover(clear_cover_mm, tie_diameter_mm, main_bar_diameter_mm)


def is_layout_feasible(
    shape: str,
    n_bars: int,
    bar_diameter_mm: float,
    clear_cover_mm: float,
    tie_diameter_mm: float,
    geometry: SectionGeometry,
) -> tuple[bool, str]:
    min_clear_spacing = max(40.0, 1.5 * bar_diameter_mm)

    if shape in {"Square", "Rectangle"}:
        b_core = usable_core_dimension(geometry.gross_width_mm, clear_cover_mm, tie_diameter_mm, bar_diameter_mm)
        h_core = usable_core_dimension(geometry.gross_depth_mm, clear_cover_mm, tie_diameter_mm, bar_diameter_mm)
        if b_core <= 0 or h_core <= 0:
            return False, "Cover/tie/main bar too large for the selected section."

        perimeter_capacity = 2.0 * (b_core + h_core)
        required_perimeter = n_bars * bar_diameter_mm + n_bars * min_clear_spacing
        if required_perimeter > perimeter_capacity:
            return False, "Insufficient perimeter to place bars with minimum clear spacing."

        if shape == "Square" and n_bars < 4:
            return False, "Square section should use at least 4 longitudinal bars."
        if shape == "Rectangle" and n_bars < 4:
            return False, "Rectangle section should use at least 4 longitudinal bars."
        return True, "OK"

    if n_bars < 6:
        return False, "Circular section should use at least 6 longitudinal bars."

    core_diameter = geometry.gross_diameter_mm - 2.0 * effective_cover(clear_cover_mm, tie_diameter_mm, bar_diameter_mm)
    if core_diameter <= 0:
        return False, "Cover/tie/main bar too large for the selected section."

    circumference = math.pi * core_diameter
    center_to_center_spacing = circumference / n_bars
    clear_spacing = center_to_center_spacing - bar_diameter_mm
    if clear_spacing < min_clear_spacing:
        return False, "Insufficient circular spacing between bars."
    return True, "OK"


def required_steel_area_mm2(
    pu_kN: float,
    phi: float,
    fc_mpa: float,
    fy_mpa: float,
    ag_mm2: float,
    min_ratio: float,
) -> tuple[float, float, float]:
    pu_n = pu_kN * 1000.0
    min_ast = min_ratio * ag_mm2
    numerator = pu_n / phi - 0.85 * fc_mpa * ag_mm2
    denominator = fy_mpa - 0.85 * fc_mpa
    calculated = numerator / denominator if denominator > 0 else 0.0
    required = max(min_ast, calculated, 0.0)
    return required, calculated, min_ast


def design_strength_kN(phi: float, fc_mpa: float, fy_mpa: float, ag_mm2: float, ast_mm2: float) -> float:
    pn_n = phi * (0.85 * fc_mpa * (ag_mm2 - ast_mm2) + fy_mpa * ast_mm2)
    return pn_n / 1000.0


def reinforcement_options(
    geometry: SectionGeometry,
    shape: str,
    required_ast_mm2: float,
    design_pu_kN: float,
    max_ratio: float,
    fy_mpa: float,
    fc_mpa: float,
    phi: float,
    clear_cover_mm: float,
    tie_diameter_mm: float,
    available_diams: list[float],
) -> pd.DataFrame:
    rows: list[dict] = []
    ag_mm2 = geometry.area_mm2
    max_ast = max_ratio * ag_mm2

    min_bars = 4 if shape in {"Square", "Rectangle"} else 6
    max_bars = 20 if shape in {"Square", "Rectangle"} else 24

    for dia in available_diams:
        area_bar = rebar_area(dia)
        for n_bars in range(min_bars, max_bars + 1):
            if shape in {"Square", "Rectangle"} and n_bars % 2 != 0:
                continue

            ast = n_bars * area_bar
            if ast < required_ast_mm2 or ast > max_ast:
                continue

            feasible, remark = is_layout_feasible(shape, n_bars, dia, clear_cover_mm, tie_diameter_mm, geometry)
            if not feasible:
                continue

            phi_pn = design_strength_kN(phi, fc_mpa, fy_mpa, ag_mm2, ast)
            reserve = phi_pn - design_pu_kN
            rows.append(
                {
                    "Bars": n_bars,
                    "Bar size (mm)": dia,
                    "Ast provided (mm²)": round(ast, 1),
                    "Steel ratio (%)": round(ast / ag_mm2 * 100.0, 3),
                    "φPn (kN)": round(phi_pn, 1),
                    "Reserve (kN)": round(reserve, 1),
                    "Comment": remark,
                }
            )

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(by=["Ast provided (mm²)", "Bars", "Bar size (mm)"]).reset_index(drop=True)
    return df


st.title("Pile Reinforcement Designer")
st.caption("ออกแบบเหล็กเสริมเสาเข็มหน้าตัดสี่เหลี่ยม สี่เหลี่ยมผืนผ้า และวงกลม สำหรับแรงอัดแกนแบบ simplified RC column check")

with st.sidebar:
    st.header("Design Basis")
    code_basis = st.selectbox("Reference basis", ["ACI-style simplified concentric compression"], index=0)
    column_type = st.radio("Transverse reinforcement type", ["Tied", "Spiral"], index=0)
    min_ratio_percent = st.number_input("Minimum steel ratio (%)", min_value=0.1, max_value=8.0, value=1.0, step=0.1)
    max_ratio_percent = st.number_input("Maximum steel ratio (%)", min_value=1.0, max_value=12.0, value=8.0, step=0.5)
    tie_dia = st.number_input("Tie / spiral diameter (mm)", min_value=6.0, max_value=20.0, value=9.0, step=1.0)
    available_bar_sizes_text = st.text_input("Available main bar diameters (mm)", value="12, 16, 20, 25, 28, 32")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Section Input")
    shape = st.selectbox("Pile cross-section", ["Square", "Rectangle", "Circular"])
    if shape == "Square":
        width = st.number_input("Width = Depth (mm)", min_value=150.0, value=300.0, step=25.0)
        depth = width
        diameter = None
    elif shape == "Rectangle":
        width = st.number_input("Width b (mm)", min_value=150.0, value=300.0, step=25.0)
        depth = st.number_input("Depth h (mm)", min_value=150.0, value=400.0, step=25.0)
        diameter = None
    else:
        diameter = st.number_input("Diameter D (mm)", min_value=150.0, value=400.0, step=25.0)
        width = None
        depth = None

    clear_cover = st.number_input("Clear cover to tie/spiral (mm)", min_value=25.0, value=50.0, step=5.0)

with col2:
    st.subheader("Material and Load Input")
    fc = st.number_input("Concrete strength f'c (MPa)", min_value=15.0, value=28.0, step=1.0)
    fy = st.number_input("Steel yield strength fy (MPa)", min_value=240.0, value=420.0, step=10.0)
    design_pu_kN = st.number_input("Required factored axial load Pu (kN)", min_value=0.0, value=1200.0, step=50.0)
    user_bar_dia = st.number_input("Trial main bar diameter for quick check (mm)", min_value=10.0, value=20.0, step=1.0)
    user_n_bars = st.number_input("Trial number of main bars", min_value=4, value=8, step=1)

geometry = section_geometry(shape, width, depth, diameter)
phi = phi_factor(column_type)
min_ratio = min_ratio_percent / 100.0
max_ratio = max_ratio_percent / 100.0

available_bar_sizes = sorted(
    {
        float(item.strip())
        for item in available_bar_sizes_text.split(",")
        if item.strip()
    }
)

required_ast, raw_ast, min_ast = required_steel_area_mm2(
    pu_kN=design_pu_kN,
    phi=phi,
    fc_mpa=fc,
    fy_mpa=fy,
    ag_mm2=geometry.area_mm2,
    min_ratio=min_ratio,
)
max_ast = max_ratio * geometry.area_mm2

trial_ast = user_n_bars * rebar_area(user_bar_dia)
trial_strength = design_strength_kN(phi, fc, fy, geometry.area_mm2, trial_ast)
trial_ok, trial_comment = is_layout_feasible(shape, int(user_n_bars), user_bar_dia, clear_cover, tie_dia, geometry)
trial_pass = trial_ok and trial_ast >= required_ast and trial_ast <= max_ast and trial_strength >= design_pu_kN

result1, result2, result3, result4 = st.columns(4)
result1.metric("Gross area Ag", f"{geometry.area_mm2:,.0f} mm²")
result2.metric("Strength reduction φ", f"{phi:.2f}")
result3.metric("Required Ast", f"{required_ast:,.1f} mm²")
result4.metric("Required steel ratio", f"{required_ast / geometry.area_mm2 * 100.0:.3f} %")

st.subheader("Quick Trial Check")
trial_data = pd.DataFrame(
    [
        {
            "Item": "Trial steel area Ast",
            "Value": f"{trial_ast:,.1f} mm²",
        },
        {
            "Item": "Trial steel ratio",
            "Value": f"{trial_ast / geometry.area_mm2 * 100.0:.3f} %",
        },
        {
            "Item": "Trial φPn",
            "Value": f"{trial_strength:,.1f} kN",
        },
        {
            "Item": "Layout check",
            "Value": "Pass" if trial_ok else f"Fail: {trial_comment}",
        },
        {
            "Item": "Overall result",
            "Value": "OK" if trial_pass else "Not adequate",
        },
    ]
)
st.table(trial_data)

st.subheader("Recommended Bar Combinations")
options_df = reinforcement_options(
    geometry=geometry,
    shape=shape,
    required_ast_mm2=required_ast,
    design_pu_kN=design_pu_kN,
    max_ratio=max_ratio,
    fy_mpa=fy,
    fc_mpa=fc,
    phi=phi,
    clear_cover_mm=clear_cover,
    tie_diameter_mm=tie_dia,
    available_diams=available_bar_sizes,
)

if options_df.empty:
    st.warning("No feasible bar combination found from the available bar diameters. Try increasing section size, using larger bars, or expanding available bar sizes.")
else:
    st.dataframe(options_df, use_container_width=True)
    best = options_df.iloc[0]
    st.success(
        f"Suggested starting option: {int(best['Bars'])} DB{int(best['Bar size (mm)'])} "
        f"(Ast = {best['Ast provided (mm²)']:,.1f} mm², φPn = {best['φPn (kN)']:,.1f} kN)"
    )

with st.expander("Calculation Details"):
    st.markdown(
        f"""
        **Selected basis**: {code_basis}

        **Geometry**
        - Shape: {shape}
        - Gross area, Ag = {geometry.area_mm2:,.1f} mm²

        **Material**
        - Concrete strength, f'c = {fc:.1f} MPa
        - Steel yield strength, fy = {fy:.1f} MPa
        - Strength reduction factor, φ = {phi:.2f}

        **Design equation**
        - φPn = φ[0.85f'c(Ag - Ast) + fyAst]
        - Required design load, Pu = {design_pu_kN:,.1f} kN

        **Steel limits**
        - Minimum Ast from ratio = {min_ast:,.1f} mm²
        - Maximum Ast from ratio = {max_ast:,.1f} mm²
        - Pure equation Ast before min check = {raw_ast:,.1f} mm²
        - Adopted required Ast = {required_ast:,.1f} mm²
        """
    )

with st.expander("Important Assumptions / Notes"):
    st.markdown(
        """
        - แอปนี้ใช้การตรวจแรงอัดแกนแบบ simplified concentric compression ของเสาคอนกรีตเสริมเหล็ก
        - ยังไม่ได้ทำ interaction diagram สำหรับแรงอัดร่วมดัด หรือ slenderness / buckling check
        - การจัดวางเหล็กใช้ spacing check แบบประมาณเพื่อคัดตัวเลือกที่วางได้เบื้องต้น
        - ผู้ใช้งานควรตรวจซ้ำตามมาตรฐานที่ใช้จริงของโครงการ เช่น ACI, มยผ., หรือมาตรฐานองค์กร
        - หากต้องการต่อยอด สามารถเพิ่ม moment input, tie spacing design, และ export report ได้
        """
    )
