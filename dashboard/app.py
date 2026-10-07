"""Four-page OULAD learning analytics dashboard.

The pages follow one analytical flow: outcomes and geography, learning
behaviour, multivariate interaction, then model evaluation. The app reads
reproducible processed artifacts; it never trains the model or joins the
multi-million-row VLE table while rendering.
"""

from __future__ import annotations

import html
import json
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard_data import (
    REGION_GEOJSON_PATH,
    compute_kpis,
    load_analysis_data,
    load_dashboard_mart,
    load_feature_snapshot,
    load_model_table,
    load_predictions,
)


PAGE_OPTIONS = [
    "Outcome & Geography",
    "Learning Behavior",
    "Interaction Analysis",
    "Prediction & Action",
]
PAGE_LABELS = {
    "Outcome & Geography": "1 · Bức tranh kết quả học tập",
    "Learning Behavior": "2 · Các yếu tố học tập",
    "Interaction Analysis": "3 · Kết hợp nhiều yếu tố",
    "Prediction & Action": "4 · Mô hình & yếu tố dự báo",
}
ATTEMPT_KEY = ["code_module", "code_presentation", "id_student"]
RESULT_ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"]
RESULT_COLORS = {
    "Distinction": "#0F766E",
    "Pass": "#2563EB",
    "Fail": "#F97316",
    "Withdrawn": "#DC2626",
}
RESULT_LABELS = {
    "Distinction": "Xuất sắc",
    "Pass": "Qua môn",
    "Fail": "Trượt",
    "Withdrawn": "Rút học",
}
RISK_COLORS = {
    "Qua môn / Xuất sắc": "#2563EB",
    "Trượt": "#DC2626",
}
EDUCATION_LABELS = {
    "No Formal quals": "Không có bằng cấp chính quy",
    "Lower Than A Level": "Dưới A Level",
    "A Level or Equivalent": "A Level hoặc tương đương",
    "HE Qualification": "Đã có bằng đại học/cao đẳng",
    "Post Graduate Qualification": "Sau đại học",
}
INK = "#0F172A"
MUTED = "#475569"
GRID = "#E2E8F0"
PANEL = "#FFFFFF"
RISK_SCALE = ["#ECFDF5", "#FEF3C7", "#FDBA74", "#EF4444", "#991B1B"]


st.set_page_config(
    page_title="OULAD Learning Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_dashboard_css() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #F8FAFC; }
        .block-container { padding-top: 1.35rem; padding-bottom: 3rem; max-width: 1520px; }
        h1, h2, h3 { color: #0F172A; letter-spacing: -0.025em; }
        p, label, .stCaption { color: #334155; }
        [data-testid="stSidebar"] { border-right: 1px solid #E2E8F0; }
        [data-testid="stMetric"] {
            background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
            padding: 0.85rem 1rem; box-shadow: 0 1px 2px rgba(15,23,42,.04);
        }
        [data-testid="stMetricLabel"] { color: #475569; }
        [data-testid="stMetricValue"] { color: #0F172A; }
        [data-testid="stPlotlyChart"] {
            background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 14px;
            padding: .3rem; box-shadow: 0 1px 3px rgba(15,23,42,.05);
        }
        .page-kicker {
            color: #2563EB; font-size: .78rem; font-weight: 800;
            letter-spacing: .08em; text-transform: uppercase; margin-bottom: .15rem;
        }
        .page-subtitle { color: #475569; font-size: 1.02rem; margin-top: -.35rem; }
        .scope-note {
            display: inline-block; background: #EFF6FF; color: #1E40AF;
            border: 1px solid #BFDBFE; border-radius: 999px; padding: .28rem .7rem;
            font-size: .82rem; font-weight: 650; margin: .2rem 0 .8rem 0;
        }
        .story-card {
            background: linear-gradient(135deg,#EFF6FF 0%,#FFFFFF 100%);
            border: 1px solid #BFDBFE; border-left: 5px solid #2563EB;
            border-radius: 12px; padding: .9rem 1.05rem; margin: .55rem 0;
        }
        .story-card h3 { margin: 0 0 .35rem 0; font-size: 1.1rem; }
        .story-card ul { margin: .15rem 0 0 1.1rem; padding: 0; }
        .story-card li { margin: .22rem 0; color: #1E293B; }
        .active-filter {
            background: #FFF7ED; color: #9A3412; border: 1px solid #FDBA74;
            border-radius: 9px; padding: .55rem .75rem; margin: .15rem 0 .65rem 0;
        }
        .definition-box {
            background: #F0FDFA; border: 1px solid #99F6E4; border-left: 5px solid #0F766E;
            border-radius: 12px; padding: .9rem 1.05rem; margin: .4rem 0 .9rem 0;
        }
        .term-guide {
            background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 12px;
            padding: .75rem 1rem; margin: .35rem 0 1rem 0; color: #334155;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def polish_figure(
    figure: go.Figure,
    *,
    height: int,
    legend: str = "top",
    hovermode: str | bool = "closest",
) -> go.Figure:
    figure.update_layout(
        template="plotly_white",
        height=height,
        margin={"l": 60, "r": 35, "t": 78, "b": 60},
        paper_bgcolor=PANEL,
        plot_bgcolor=PANEL,
        font={"family": "Arial, sans-serif", "size": 13, "color": INK},
        title={"x": .02, "xanchor": "left", "font": {"size": 19, "color": INK}},
        hoverlabel={"bgcolor": "#FFFFFF", "font_size": 13, "font_color": INK},
        hovermode=hovermode,
    )
    figure.update_xaxes(
        showgrid=False,
        linecolor=GRID,
        tickfont={"color": MUTED},
        title_font={"color": MUTED},
        automargin=True,
    )
    figure.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=GRID,
        tickfont={"color": MUTED},
        title_font={"color": MUTED},
        automargin=True,
    )
    if legend == "top":
        figure.update_layout(
            legend={
                "orientation": "h",
                "yanchor": "bottom",
                "y": 1.01,
                "xanchor": "right",
                "x": 1,
                "title_text": "",
                "bgcolor": "rgba(255,255,255,.88)",
            }
        )
    elif legend == "none":
        figure.update_layout(showlegend=False)
    return figure


def render_header(kicker: str, title: str, subtitle: str, scope: str) -> None:
    st.markdown(
        f'<div class="page-kicker">{html.escape(kicker)}</div>',
        unsafe_allow_html=True,
    )
    st.title(title)
    st.markdown(
        f'<div class="page-subtitle">{html.escape(subtitle)}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="scope-note">{html.escape(scope)}</div>',
        unsafe_allow_html=True,
    )


def render_term_guide(items: list[tuple[str, str]]) -> None:
    """Show important abbreviations and anonymised codes in the reading flow."""
    content = " · ".join(
        f"<b>{html.escape(term)}</b>: {html.escape(definition)}"
        for term, definition in items
    )
    st.markdown(
        f'<div class="term-guide"><b>Đọc nhanh thuật ngữ</b><br>{content}</div>',
        unsafe_allow_html=True,
    )


def render_story(title: str, bullets: list[str]) -> None:
    items = "".join(f"<li>{html.escape(item)}</li>" for item in bullets)
    st.markdown(
        f'<div class="story-card"><h3>{html.escape(title)}</h3><ul>{items}</ul></div>',
        unsafe_allow_html=True,
    )


def format_percent(value: float) -> str:
    return "—" if pd.isna(value) else f"{value:.1%}"


def chart_points(event: Any) -> list[Any]:
    if event is None:
        return []
    try:
        return list(event.selection.points)
    except (AttributeError, KeyError, TypeError):
        try:
            return list(event["selection"]["points"])
        except (KeyError, TypeError):
            return []


def point_value(point: Any, field: str) -> Any:
    try:
        return point.get(field)
    except AttributeError:
        try:
            return point[field]
        except (KeyError, TypeError):
            return None


@st.cache_data(show_spinner=False)
def analysis_data() -> pd.DataFrame:
    return load_analysis_data()


@st.cache_data(show_spinner=False)
def feature_snapshot() -> pd.DataFrame:
    return load_feature_snapshot()


@st.cache_data(show_spinner=False)
def predictions() -> pd.DataFrame:
    return load_predictions()


@st.cache_data(show_spinner=False)
def model_table(filename: str) -> pd.DataFrame:
    return load_model_table(filename)


@st.cache_data(show_spinner=False)
def dashboard_mart(filename: str, required: tuple[str, ...]) -> pd.DataFrame:
    return load_dashboard_mart(filename, set(required))


def filter_context(
    frame: pd.DataFrame,
    genders: list[str],
    age_bands: list[str],
    education_levels: list[str],
    imd_bands: list[str],
    regions: list[str],
) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for column, values in (
        ("gender", genders),
        ("age_band", age_bands),
        ("highest_education", education_levels),
        ("imd_band", imd_bands),
        ("region", regions),
    ):
        if values and column in frame:
            mask &= frame[column].isin(values)
    return frame.loc[mask].copy()


def page_navigation() -> str:
    st.sidebar.title("Phân tích kết quả học tập")
    st.sidebar.caption(
        "Đề tài: Nghiên cứu và phân tích các yếu tố ảnh hưởng đến kết quả học tập của sinh viên đại học"
    )
    st.sidebar.caption("Dữ liệu OULAD · Python · Streamlit · Plotly")
    page = st.sidebar.radio(
        "Điều hướng",
        PAGE_OPTIONS,
        format_func=lambda value: PAGE_LABELS[value],
    )
    st.sidebar.divider()
    with st.sidebar.expander("Thuật ngữ dùng trong dashboard", expanded=True):
        st.markdown(
            "- **OULAD:** Open University Learning Analytics Dataset.\n"
            "- **VLE:** hệ thống học trực tuyến; số click chỉ phản ánh mức độ sử dụng hệ thống.\n"
            "- **IMD:** nhóm mức khó khăn kinh tế–xã hội của khu vực cư trú.\n"
            "- **Trượt học phần:** kết quả cuối là Fail; không gộp sinh viên rút học.\n"
            "- **Rút học:** kết quả Withdrawn; được mô tả riêng và không đưa vào model.\n"
            "- **Lượt học:** một lần đăng ký học; một sinh viên có thể xuất hiện nhiều lần.\n"
            "- **Bài tập/kiểm tra (assessment):** hoạt động được chấm điểm.\n"
            "- **Cảnh báo giữa khóa ngày 105:** chỉ dùng dữ liệu có sẵn đến ngày 105 để dự đoán nguy cơ trượt.\n"
            "- Kết quả thể hiện liên hệ, không khẳng định nhân quả."
        )
    return page


def imd_sort_key(value: str) -> tuple[int, str]:
    if value == "Không xác định":
        return (999, value)
    try:
        return (int(value.split("-")[0].replace("%", "")), value)
    except ValueError:
        return (998, value)


def context_filters(
    frame: pd.DataFrame,
    *,
    key_prefix: str,
) -> tuple[list[str], list[str], list[str], list[str]]:
    st.subheader("Bộ lọc theo đặc điểm người học")
    columns = st.columns(4)
    genders = columns[0].multiselect(
        "Giới tính · gender",
        sorted(frame["gender"].dropna().unique()),
        key=f"{key_prefix}_genders",
        placeholder="Tất cả giới tính",
        format_func=lambda value: {"F": "Nữ", "M": "Nam"}.get(value, value),
    )
    age_order = [
        value
        for value in ["0-35", "35-55", "55<="]
        if value in set(frame["age_band"].dropna())
    ]
    age_bands = columns[1].multiselect(
        "Nhóm tuổi",
        age_order,
        key=f"{key_prefix}_age_bands",
        placeholder="Tất cả nhóm tuổi",
    )
    education_order = [
        value
        for value in EDUCATION_LABELS
        if value in set(frame["highest_education"].dropna())
    ]
    education_levels = columns[2].multiselect(
        "Học vấn trước khi nhập học",
        education_order,
        key=f"{key_prefix}_education",
        placeholder="Tất cả trình độ",
        format_func=lambda value: EDUCATION_LABELS.get(value, value),
    )
    imd_options = sorted(
        frame["imd_band"].dropna().astype(str).unique(), key=imd_sort_key
    )
    imd_bands = columns[3].multiselect(
        "Mức điều kiện kinh tế–xã hội khu vực",
        imd_options,
        key=f"{key_prefix}_imd",
        placeholder="Tất cả nhóm khu vực",
    )
    return genders, age_bands, education_levels, imd_bands


def render_academic_kpis(frame: pd.DataFrame) -> None:
    kpis = compute_kpis(frame)
    pass_rate = frame["final_result"].isin(["Pass", "Distinction"]).mean()
    cards = st.columns(4)
    cards[0].metric("Tổng sinh viên", f"{kpis.learners:,}")
    cards[1].metric("Điểm trung bình", f"{kpis.average_assessment_score:.1f}")
    cards[2].metric("Tỷ lệ qua môn", format_percent(pass_rate))
    cards[3].metric("Tỷ lệ trượt", format_percent(kpis.at_risk_rate))


def render_region_map(base_frame: pd.DataFrame, active_region: str | None) -> None:
    stats = (
        base_frame.groupby("region", observed=True)
        .agg(
            attempts=("id_student", "size"),
            at_risk_count=("At_Risk", "sum"),
            at_risk_rate=("At_Risk", "mean"),
        )
        .reset_index()
    )
    with REGION_GEOJSON_PATH.open("r", encoding="utf-8") as handle:
        geojson = json.load(handle)
    color_min = float(stats["at_risk_rate"].min())
    color_max = float(stats["at_risk_rate"].max())
    if np.isclose(color_min, color_max):
        color_min = max(0.0, color_min - .01)
        color_max = min(1.0, color_max + .01)
    figure = px.choropleth(
        stats,
        geojson=geojson,
        locations="region",
        featureidkey="properties.region",
        color="at_risk_rate",
        color_continuous_scale=RISK_SCALE,
        range_color=(color_min, color_max),
        custom_data=["region", "attempts", "at_risk_count", "at_risk_rate"],
        title="2 · Tỷ lệ trượt theo vùng cư trú",
    )
    figure.update_geos(fitbounds="locations", visible=False, bgcolor="#FFFFFF")
    figure.update_traces(
        marker_line_color="#FFFFFF",
        marker_line_width=1.0,
        hovertemplate=(
            "Vùng=%{customdata[0]}<br>Tỷ lệ trượt=%{customdata[3]:.1%}"
            "<br>Lượt học=%{customdata[1]:,}<br>Số lượt trượt=%{customdata[2]:,}<extra></extra>"
        ),
    )
    figure.update_layout(
        coloraxis_colorbar={"title": "Tỷ lệ trượt", "tickformat": ".0%"}
    )
    polish_figure(figure, height=565, legend="none")
    event = st.plotly_chart(
        figure,
        width="stretch",
        on_select="rerun",
        selection_mode="points",
        key=f"academic_region_map_{st.session_state.get('map_key_version', 0)}",
    )
    points = chart_points(event)
    if points:
        selected = point_value(points[0], "location")
        if selected is None:
            custom = point_value(points[0], "customdata")
            selected = custom[0] if custom else None
        if selected and str(selected) != active_region:
            st.session_state["academic_region"] = str(selected)
            st.rerun()
    st.caption(
        "Bấm một vùng để lọc KPI và các biểu đồ trên Trang 1. "
        "Màu đậm hơn = tỷ lệ trượt cao hơn; tooltip cho biết tỷ lệ và cỡ mẫu N."
    )


def outcome_percentages(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    counts = (
        frame.groupby([group_column, "final_result"], observed=True)
        .size()
        .rename("attempts")
        .reset_index()
    )
    counts["share"] = counts["attempts"] / counts.groupby(group_column)[
        "attempts"
    ].transform("sum")
    counts["result_label"] = counts["final_result"].map(RESULT_LABELS)
    return counts


def render_outcome_overview(frame: pd.DataFrame) -> None:
    detail = bool(st.session_state.get("outcome_education_detail", False))
    controls = st.columns([4, 1])
    controls[0].markdown(
        "**Cơ cấu chung**" if not detail else "**Phân tích sâu theo học vấn đầu vào**"
    )
    button_label = "Xem theo học vấn đầu vào" if not detail else "Quay lại tổng thể"
    if controls[1].button(
        button_label,
        key="toggle_outcome_education",
        width="stretch",
    ):
        st.session_state["outcome_education_detail"] = not detail
        st.rerun()

    chart_frame = frame.copy()
    if detail:
        chart_frame["Nhóm"] = chart_frame["highest_education"].map(
            EDUCATION_LABELS
        ).fillna("Không xác định")
        group_order = (
            chart_frame.groupby("Nhóm", observed=True)["At_Risk"]
            .mean()
            .sort_values()
            .index.tolist()
        )
        title = "1 · Cơ cấu kết quả theo học vấn trước khi nhập học"
        height = 500
    else:
        chart_frame["Nhóm"] = "Toàn bộ sinh viên"
        group_order = ["Toàn bộ sinh viên"]
        title = "1 · Cơ cấu kết quả học tập tổng thể"
        height = 330

    data = outcome_percentages(chart_frame, "Nhóm")
    data["inside_label"] = data.apply(
        lambda row: f"{row['share']:.0%}<br>N={int(row['attempts']):,}"
        if row["share"] >= .06
        else "",
        axis=1,
    )
    figure = px.bar(
        data,
        x="share",
        y="Nhóm",
        color="result_label",
        orientation="h",
        barmode="stack",
        category_orders={
            "result_label": [RESULT_LABELS[value] for value in RESULT_ORDER],
            "Nhóm": group_order,
        },
        color_discrete_map={
            RESULT_LABELS[key]: value for key, value in RESULT_COLORS.items()
        },
        custom_data=["attempts"],
        text="inside_label",
        labels={"share": "Tỷ trọng", "result_label": "Kết quả"},
        title=title,
    )
    figure.update_traces(
        texttemplate="%{text}",
        textposition="inside",
        hovertemplate=(
            "Nhóm=%{y}<br>Kết quả=%{fullData.name}<br>Tỷ trọng=%{x:.1%}"
            "<br>N=%{customdata[0]:,}<extra></extra>"
        ),
    )
    figure.update_xaxes(tickformat=".0%", range=[0, 1])
    figure.update_yaxes(categoryorder="array", categoryarray=group_order)
    polish_figure(figure, height=height)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Mỗi thanh bằng 100%; tỷ lệ và N được ghi trực tiếp. "
        "Nút phân tích sâu chuyển từ tổng thể sang một yếu tố có ý nghĩa diễn giải."
    )


def render_vle_timeline(
    frame: pd.DataFrame,
    genders: list[str],
    age_bands: list[str],
    education_levels: list[str],
    imd_bands: list[str],
    regions: list[str],
) -> tuple[float, float]:
    daily = dashboard_mart(
        "vle_daily_profile.csv.gz",
        (
            "code_module",
            "code_presentation",
            "gender",
            "age_band",
            "highest_education",
            "imd_band",
            "region",
            "At_Risk",
            "date",
            "sum_click",
        ),
    )
    daily = filter_context(
        daily, genders, age_bands, education_levels, imd_bands, regions
    )
    daily = daily.loc[daily["date"].le(105)].copy()
    totals = frame.groupby("At_Risk").size().rename("attempts")
    daily = daily.groupby(["At_Risk", "date"], as_index=False)["sum_click"].sum()
    if daily.empty:
        st.info("Không có sự kiện VLE trong phạm vi lọc.")
        return float("nan"), float("nan")

    dates = np.arange(int(daily["date"].min()), int(daily["date"].max()) + 1)
    statuses = sorted(totals.index.tolist())
    grid = pd.MultiIndex.from_product(
        [statuses, dates], names=["At_Risk", "date"]
    ).to_frame(index=False)
    daily = grid.merge(daily, on=["At_Risk", "date"], how="left").fillna(
        {"sum_click": 0}
    )
    daily = daily.merge(totals, on="At_Risk", how="left")
    daily["avg_sum_click"] = daily["sum_click"] / daily["attempts"]
    daily = daily.sort_values(["At_Risk", "date"])
    daily["avg_click_7d"] = daily.groupby("At_Risk")["avg_sum_click"].transform(
        lambda values: values.rolling(7, min_periods=1).mean()
    )
    daily["Nhóm"] = daily["At_Risk"].map(
        {0: "Qua môn / Xuất sắc", 1: "Trượt"}
    )

    figure = px.line(
        daily,
        x="date",
        y="avg_click_7d",
        color="Nhóm",
        category_orders={
            "Nhóm": ["Qua môn / Xuất sắc", "Trượt"]
        },
        color_discrete_map=RISK_COLORS,
        labels={
            "date": "Ngày tương đối từ khi môn học bắt đầu",
            "avg_click_7d": "Lượt tương tác trung bình / lượt học / ngày",
        },
        title="4 · Mức tham gia học trực tuyến theo thời gian",
    )
    figure.update_traces(
        line={"width": 3},
        hovertemplate="Ngày=%{x}<br>Trung bình 7 ngày=%{y:.2f} lượt tương tác<extra></extra>",
    )

    deadlines = dashboard_mart(
        "assessment_deadlines.csv",
        (
            "code_module",
            "code_presentation",
            "assessment_type",
            "due_date",
            "weight",
        ),
    )
    deadlines = deadlines.loc[deadlines["due_date"].le(105)].copy()
    milestones = (
        deadlines.dropna(subset=["due_date"])
        .groupby("due_date", as_index=False)["weight"]
        .sum()
        .sort_values("weight", ascending=False)
        .head(3)
        .sort_values("due_date")
    )
    for row in milestones.itertuples(index=False):
        figure.add_vline(
            x=float(row.due_date),
            line_dash="dot",
            line_color="#64748B",
            line_width=1.2,
        )
    polish_figure(figure, height=505, hovermode="x unified")
    st.plotly_chart(figure, width="stretch")
    milestone_days = ", ".join(str(int(value)) for value in milestones["due_date"])
    st.caption(
        "Đường là trung bình trượt 7 ngày; mẫu số gồm mọi lượt học trong từng nhóm, kể cả ngày không click. "
        f"Vạch chấm đánh dấu ba hạn nộp trọng số lớn nhất: ngày {milestone_days}."
    )
    means = frame.groupby("At_Risk")["vle_total_clicks_all_time"].mean()
    return float(means.get(1, np.nan)), float(means.get(0, np.nan))


def render_submission_scatter(
    genders: list[str],
    age_bands: list[str],
    education_levels: list[str],
    imd_bands: list[str],
    regions: list[str],
) -> float:
    submissions = dashboard_mart(
        "assessment_submissions.csv.gz",
        (
            "code_module",
            "code_presentation",
            "date_submitted",
            "gender",
            "age_band",
            "highest_education",
            "imd_band",
            "region",
            "num_of_prev_attempts",
            "At_Risk",
            "submission_delay",
            "score",
        ),
    )
    submissions = filter_context(
        submissions, genders, age_bands, education_levels, imd_bands, regions
    )
    submissions = submissions.loc[submissions["date_submitted"].le(105)].dropna(
        subset=["submission_delay", "score"]
    )
    if submissions.empty:
        st.info("Không có bài nộp có đủ ngày hạn và điểm trong phạm vi lọc.")
        return float("nan")

    sample = submissions.sample(min(4500, len(submissions)), random_state=42)
    figure = go.Figure()
    for at_risk, label in (
        (0, "Qua môn / Xuất sắc"),
        (1, "Trượt"),
    ):
        group = sample.loc[sample["At_Risk"].eq(at_risk)]
        figure.add_trace(
            go.Scattergl(
                x=group["submission_delay"],
                y=group["score"],
                mode="markers",
                name=label,
                customdata=group[["num_of_prev_attempts"]],
                marker={
                    "size": 7 + 2 * group["num_of_prev_attempts"].clip(upper=6),
                    "color": RISK_COLORS[label],
                    "opacity": .42,
                    "line": {"color": "#FFFFFF", "width": .4},
                },
                hovertemplate=(
                    "Nộp trễ=%{x:.0f} ngày<br>Điểm=%{y:.1f}"
                    "<br>Số lần từng học=%{customdata[0]:.0f}<extra></extra>"
                ),
            )
        )
    correlation = float(
        submissions[["submission_delay", "score"]].corr().iloc[0, 1]
    )
    if submissions["submission_delay"].nunique() > 1:
        slope, intercept = np.polyfit(
            submissions["submission_delay"], submissions["score"], 1
        )
        x_line = np.array(
            [
                submissions["submission_delay"].min(),
                submissions["submission_delay"].max(),
            ]
        )
        figure.add_trace(
            go.Scatter(
                x=x_line,
                y=intercept + slope * x_line,
                mode="lines",
                name="Trendline toàn bộ",
                line={"color": INK, "width": 3, "dash": "dash"},
                hoverinfo="skip",
            )
        )
    figure.add_vline(
        x=0,
        line_color="#64748B",
        line_dash="dot",
    )
    figure.update_layout(
        title="6 · Nộp bài sớm hoặc trễ liên quan thế nào đến điểm số?",
        xaxis_title="Số ngày nộp trễ (âm = nộp sớm)",
        yaxis_title="Điểm bài tập/kiểm tra",
    )
    figure.update_yaxes(range=[-2, 102])
    polish_figure(figure, height=535)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"Đường xu hướng dùng {len(submissions):,} bài đã nộp trước hoặc tại ngày 105; "
        f"đồ thị lấy mẫu cố định {len(sample):,} điểm để dễ đọc. Vạch dọc tại 0 là đúng hạn; "
        "kích thước điểm tăng theo số lần học trước."
    )
    return correlation


def render_activity_comparison(
    genders: list[str],
    age_bands: list[str],
    education_levels: list[str],
    imd_bands: list[str],
    regions: list[str],
) -> tuple[str, float]:
    activity = dashboard_mart(
        "vle_activity_summary_day105.csv.gz",
        (
            "code_module",
            "code_presentation",
            "gender",
            "age_band",
            "highest_education",
            "imd_band",
            "region",
            "activity_type",
            "At_Risk",
            "sum_click",
        ),
    )
    activity = filter_context(
        activity, genders, age_bands, education_levels, imd_bands, regions
    )
    activity = activity.groupby(
        ["At_Risk", "activity_type"], as_index=False
    )["sum_click"].sum()
    if activity.empty:
        st.info("Không có click VLE trong phạm vi lọc.")
        return "—", float("nan")
    activity["share"] = activity["sum_click"] / activity.groupby("At_Risk")[
        "sum_click"
    ].transform("sum")
    activity["Nhóm kết quả"] = activity["At_Risk"].map(
        {0: "Qua môn / Xuất sắc", 1: "Trượt"}
    )
    top_types = (
        activity.groupby("activity_type")["sum_click"]
        .sum()
        .nlargest(8)
        .index
    )
    display = activity.loc[activity["activity_type"].isin(top_types)].copy()
    type_order = (
        display.groupby("activity_type")["share"].max().sort_values().index.tolist()
    )
    figure = px.bar(
        display,
        x="share",
        y="activity_type",
        color="Nhóm kết quả",
        orientation="h",
        barmode="group",
        category_orders={
            "activity_type": type_order,
            "Nhóm kết quả": ["Qua môn / Xuất sắc", "Trượt"],
        },
        color_discrete_map=RISK_COLORS,
        custom_data=["sum_click"],
        text="share",
        labels={
            "share": "Tỷ trọng trong tổng tương tác của mỗi nhóm",
            "activity_type": "Loại tài nguyên VLE",
        },
        title="7 · Hai nhóm sử dụng tài nguyên học trực tuyến khác nhau thế nào?",
    )
    figure.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
        hovertemplate=(
            "Loại=%{y}<br>Nhóm=%{fullData.name}<br>Tỷ trọng=%{x:.2%}"
            "<br>Clicks=%{customdata[0]:,}<extra></extra>"
        ),
    )
    figure.update_xaxes(tickformat=".0%")
    polish_figure(figure, height=570)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Tỷ trọng được chuẩn hóa riêng trong từng nhóm kết quả, nhờ đó biểu đồ so sánh "
        "cách phân bổ hoạt động thay vì chỉ cho biết loại tài nguyên nào có nhiều click nhất."
    )
    pivot = activity.pivot(
        index="activity_type", columns="At_Risk", values="share"
    ).fillna(0)
    pivot["gap"] = pivot.get(1, 0) - pivot.get(0, 0)
    largest = pivot["gap"].abs().idxmax()
    return str(largest), float(pivot.loc[largest, "gap"])


def comparison_message(
    frame: pd.DataFrame,
    group_column: str,
    label: str,
    display_labels: dict[str, str] | None = None,
) -> str:
    grouped = (
        frame.groupby(group_column, observed=True)["At_Risk"]
        .agg(["mean", "size"])
        .sort_values("mean")
    )
    if grouped.empty:
        return f"Không đủ dữ liệu để so sánh theo {label}."
    if len(grouped) == 1:
        row = grouped.iloc[0]
        group = grouped.index[0]
        display = (display_labels or {}).get(str(group), str(group))
        return (
            f"{label} {display} có tỷ lệ trượt {row['mean']:.1%} "
            f"(N={int(row['size']):,})."
        )
    low, high = grouped.iloc[0], grouped.iloc[-1]
    high_group = str(grouped.index[-1])
    low_group = str(grouped.index[0])
    high_display = (display_labels or {}).get(high_group, high_group)
    low_display = (display_labels or {}).get(low_group, low_group)
    return (
        f"Tỷ lệ trượt theo {label.lower()} cao nhất ở {high_display} "
        f"({high['mean']:.1%}, N={int(high['size']):,}) và thấp nhất ở "
        f"{low_display} ({low['mean']:.1%}, N={int(low['size']):,})."
    )


def render_score_distribution(frame: pd.DataFrame) -> None:
    """Show the academic score distribution without hiding missing scores."""
    data = frame.dropna(subset=["assessment_score_mean_all_time"]).copy()
    if data.empty:
        st.info("Không có điểm bài tập/kiểm tra trong phạm vi lọc.")
        return
    data["Kết quả"] = data["final_result"].map(RESULT_LABELS)
    result_order = [RESULT_LABELS[value] for value in RESULT_ORDER]
    figure = px.violin(
        data,
        x="Kết quả",
        y="assessment_score_mean_all_time",
        color="Kết quả",
        category_orders={"Kết quả": result_order},
        color_discrete_map={
            RESULT_LABELS[key]: value for key, value in RESULT_COLORS.items()
        },
        box=True,
        points=False,
        labels={
            "assessment_score_mean_all_time": "Điểm bài tập/kiểm tra trung bình",
        },
        title="3 · Điểm quá trình phân bố thế nào trong từng kết quả cuối?",
    )
    figure.update_traces(
        hovertemplate="Kết quả=%{x}<br>Điểm=%{y:.1f}<extra></extra>"
    )
    figure.update_yaxes(range=[0, 100])
    polish_figure(figure, height=500, legend="none")
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"Có {len(data):,}/{len(frame):,} lượt học có ít nhất một điểm được chấm. "
        "Bề rộng thể hiện nơi dữ liệu tập trung; hộp bên trong thể hiện trung vị và khoảng 25%–75%."
    )


def render_overview_page() -> None:
    render_header(
        "Trang 1 · Bức tranh kết quả học tập",
        "Sinh viên đang đạt kết quả như thế nào?",
        "Bắt đầu từ cơ cấu tổng thể, sau đó xem kết quả thay đổi theo vùng và nền tảng đầu vào.",
        "Phạm vi mô tả toàn khóa · một dòng = một lượt đăng ký học",
    )
    render_term_guide(
        [
            ("Xuất sắc", "Distinction, mức kết quả cao hơn Qua môn"),
            ("Trượt học phần", "kết quả cuối là Fail; rút học được tách riêng"),
            ("Lượt học", "một lần đăng ký học; một sinh viên có thể xuất hiện nhiều lần"),
        ]
    )
    frame = analysis_data()
    genders, age_bands, education_levels, imd_bands = context_filters(
        frame, key_prefix="overview"
    )
    base = filter_context(
        frame, genders, age_bands, education_levels, imd_bands, []
    )
    if base.empty:
        st.warning("Bộ lọc hiện tại không có lượt học.")
        return

    active_region = st.session_state.get("academic_region")
    valid_regions = set(base["region"].dropna().astype(str))
    if active_region not in valid_regions:
        active_region = None
        st.session_state["academic_region"] = None
    effective = filter_context(
        base,
        [],
        [],
        [],
        [],
        [active_region] if active_region else [],
    )

    if active_region:
        notice = st.columns([4, 1])
        notice[0].markdown(
            f'<div class="active-filter"><b>Cross-filter từ bản đồ:</b> {html.escape(active_region)}</div>',
            unsafe_allow_html=True,
        )
        if notice[1].button(
            "Bỏ lọc vùng", key="clear_academic_region", width="stretch"
        ):
            st.session_state["academic_region"] = None
            st.session_state["map_key_version"] = (
                st.session_state.get("map_key_version", 0) + 1
            )
            st.rerun()

    render_academic_kpis(effective)
    st.caption(
        f"Phạm vi hiện tại: {len(effective):,} lượt học · "
        f"{effective['id_student'].nunique():,} sinh viên"
        + (f" · vùng {active_region}" if active_region else " · tất cả vùng")
    )
    fail_rate = float(effective["At_Risk"].mean())
    withdrawn_rate = float(effective["final_result"].eq("Withdrawn").mean())
    render_story(
        "STORY · Nhận định chính",
        [
            f"Cứ 100 lượt học thì khoảng {fail_rate * 100:.0f} lượt trượt; rút học là một kết quả khác và chiếm {withdrawn_rate:.1%}.",
            comparison_message(
                effective,
                "highest_education",
                "Trình độ trước khi nhập học",
                EDUCATION_LABELS,
            ),
            comparison_message(base, "region", "Vùng cư trú")
            + " Bản đồ giúp xác định nơi chênh lệch tập trung để đối chiếu thêm với điều kiện kinh tế–xã hội.",
        ],
    )

    st.subheader("Kết quả tổng thể và sự khác biệt theo bối cảnh")
    render_outcome_overview(effective)
    render_region_map(base, active_region)
    render_score_distribution(effective)


def filter_snapshot_context(
    frame: pd.DataFrame,
    genders: list[str],
    age_bands: list[str],
    education_levels: list[str],
    imd_bands: list[str],
) -> pd.DataFrame:
    return filter_context(
        frame, genders, age_bands, education_levels, imd_bands, []
    )


def add_behavior_bands(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    labels = ["25% thấp nhất", "Nhóm 2", "Nhóm 3", "25% cao nhất"]
    data["engagement_quartile"] = pd.qcut(
        data["vle_total_clicks_cutoff"].rank(method="first"),
        q=4,
        labels=labels,
    )
    completion = data["assessment_completion_rate_cutoff"]
    data["completion_band"] = np.select(
        [
            completion.eq(0),
            completion.gt(0) & completion.le(.5),
            completion.gt(.5) & completion.lt(1),
            completion.eq(1),
        ],
        ["0%", "Trên 0% đến 50%", "Trên 50% đến dưới 100%", "100%"],
        default="Không xác định",
    )
    return data


def render_completion_chart(frame: pd.DataFrame) -> tuple[float, float]:
    order = ["0%", "Trên 0% đến 50%", "Trên 50% đến dưới 100%", "100%"]
    grouped = (
        frame.groupby("completion_band", observed=True)
        .agg(at_risk_rate=("At_Risk", "mean"), attempts=("id_student", "size"))
        .reindex(order)
        .dropna(subset=["at_risk_rate"])
        .reset_index()
    )
    figure = px.bar(
        grouped,
        x="completion_band",
        y="at_risk_rate",
        color="at_risk_rate",
        color_continuous_scale=RISK_SCALE,
        range_color=(0, 1),
        custom_data=["attempts"],
        text="at_risk_rate",
        labels={
            "completion_band": "Mức hoàn thành bài đã đến hạn",
            "at_risk_rate": "Tỷ lệ trượt",
        },
        title="5 · Hoàn thành bài tập và tỷ lệ trượt",
    )
    figure.update_traces(
        texttemplate="%{text:.1%}",
        textposition="outside",
        hovertemplate=(
            "Mức hoàn thành=%{x}<br>Tỷ lệ trượt=%{y:.1%}"
            "<br>N=%{customdata[0]:,}<extra></extra>"
        ),
    )
    figure.update_yaxes(tickformat=".0%", range=[0, 1.08])
    figure.update_layout(coloraxis_showscale=False)
    polish_figure(figure, height=470, legend="none")
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Chỉ tính những bài tập/kiểm tra đã đến hạn trước hoặc tại ngày 105. "
        "Biểu đồ cho thấy mối liên hệ, không khẳng định đây là nguyên nhân duy nhất."
    )
    indexed = grouped.set_index("completion_band")["at_risk_rate"]
    return float(indexed.get("0%", np.nan)), float(indexed.get("100%", np.nan))


def render_behavior_page() -> None:
    render_header(
        "Trang 2 · Các yếu tố học tập",
        "Yếu tố học tập nào liên quan rõ nhất đến kết quả?",
        "So sánh mức tham gia học trực tuyến, tiến độ làm bài và thời điểm nộp bài.",
        "Chỉ dùng dữ liệu có đến ngày 105; biểu đồ cho thấy mối liên hệ, không khẳng định nguyên nhân",
    )
    render_term_guide(
        [
            ("VLE", "hệ thống học trực tuyến của trường"),
            ("Lượt tương tác", "số lần sử dụng hệ thống, không phải thời gian học hay điểm danh"),
            ("Bài đánh giá", "bài tập hoặc bài kiểm tra được chấm điểm"),
            ("Ngày 105", "mốc dữ liệu được dùng cho cảnh báo giữa khóa"),
        ]
    )
    frame = analysis_data()
    genders, age_bands, education_levels, imd_bands = context_filters(
        frame, key_prefix="behavior"
    )
    filtered = filter_context(
        frame, genders, age_bands, education_levels, imd_bands, []
    )
    snapshot = filter_snapshot_context(
        feature_snapshot(), genders, age_bands, education_levels, imd_bands
    )
    if filtered.empty or snapshot.empty:
        st.warning("Bộ lọc hiện tại không có đủ lượt học cho phân tích hành vi.")
        return
    snapshot = add_behavior_bands(snapshot)

    engagement = snapshot.groupby("engagement_quartile", observed=True)["At_Risk"].mean()
    low_click_rate = float(engagement.iloc[0])
    high_click_rate = float(engagement.iloc[-1])
    completion = snapshot.groupby("completion_band", observed=True)["At_Risk"].mean()
    zero_rate = float(completion.get("0%", np.nan))
    full_rate = float(completion.get("100%", np.nan))
    render_story(
        "STORY · Hai yếu tố liên quan rõ nhất",
        [
            f"Hoàn thành bài là yếu tố phân biệt rõ nhất: nhóm chưa hoàn thành bài đến hạn có tỷ lệ trượt {format_percent(zero_rate)}, còn nhóm hoàn thành đủ là {format_percent(full_rate)}.",
            f"Mức tham gia học trực tuyến cũng liên quan rõ: nhóm 25% ít tương tác nhất có tỷ lệ trượt {low_click_rate:.1%}, so với {high_click_rate:.1%} ở nhóm 25% tương tác nhiều nhất.",
            "Vì vậy, nên ưu tiên hỗ trợ sinh viên vừa chưa hoàn thành bài đến hạn vừa ít tham gia hệ thống học trực tuyến.",
        ],
    )

    render_vle_timeline(
        filtered, genders, age_bands, education_levels, imd_bands, []
    )
    render_completion_chart(snapshot)
    correlation = render_submission_scatter(
        genders, age_bands, education_levels, imd_bands, []
    )
    activity_type, activity_gap = render_activity_comparison(
        genders, age_bands, education_levels, imd_bands, []
    )
    delay_message = (
        "nộp càng trễ thường đi cùng điểm thấp hơn"
        if pd.notna(correlation) and correlation < 0
        else "chưa thấy xu hướng rõ giữa thời điểm nộp và điểm"
    )
    activity_group = (
        "nhóm Trượt"
        if activity_gap > 0
        else "nhóm Qua môn/Xuất sắc"
    )
    st.info(
        f"Đọc thêm: {delay_message}. Với tài nguyên `{activity_type}`, {activity_group} "
        f"có tỷ trọng sử dụng cao hơn {abs(activity_gap):.1%}. Chênh lệch loại tài nguyên "
        "nhỏ hơn nhiều so với chênh lệch về hoàn thành bài và mức hoạt động, nên loại tài nguyên là yếu tố phụ."
    )


def prepare_risk_frame() -> pd.DataFrame:
    prediction = predictions().loc[
        lambda data: data["dataset_split"].eq("test")
    ].copy()
    snapshot = feature_snapshot()[
        ATTEMPT_KEY
        + [
            "highest_education",
            "num_of_prev_attempts",
            "assessment_weighted_score_cutoff",
            "assessment_completion_rate_cutoff",
            "assessment_missed_due_count",
            "vle_active_days_last_28_days",
            "vle_days_since_last_activity",
            "vle_total_clicks_cutoff",
        ]
    ]
    frame = prediction.merge(
        snapshot, on=ATTEMPT_KEY, how="left", validate="one_to_one"
    )
    frame["imd_band"] = (
        frame["imd_band"].replace({"10-20": "10-20%"}).fillna("Không xác định")
    )
    frame["highest_education"] = frame["highest_education"].fillna(
        "Không xác định"
    )
    frame["risk_level"] = frame["risk_band"]
    return frame


def prepare_interaction_frame() -> pd.DataFrame:
    columns = ATTEMPT_KEY + [
        "gender",
        "age_band",
        "highest_education",
        "imd_band",
        "num_of_prev_attempts",
        "At_Risk",
        "vle_total_clicks_cutoff",
        "assessment_completion_rate_cutoff",
        "assessment_weighted_score_cutoff",
    ]
    frame = add_behavior_bands(feature_snapshot()[columns])
    labels = ["25% thấp nhất", "Nhóm 2", "Nhóm 3", "25% cao nhất"]
    score = frame["assessment_weighted_score_cutoff"]
    has_score = score.notna()
    frame["assessment_score_quartile"] = "Chưa có assessment được chấm"
    frame.loc[has_score, "assessment_score_quartile"] = pd.qcut(
        score.loc[has_score].rank(method="first"),
        q=4,
        labels=labels,
    ).astype(str)
    frame["actual_at_risk"] = frame["At_Risk"].astype(int)
    frame["imd_band"] = (
        frame["imd_band"].replace({"10-20": "10-20%"}).fillna("Không xác định")
    )
    frame["highest_education"] = frame["highest_education"].fillna(
        "Không xác định"
    )
    return frame


def render_engagement_assessment_heatmap(
    frame: pd.DataFrame,
) -> tuple[float, int, float, int]:
    row_order = ["25% thấp nhất", "Nhóm 2", "Nhóm 3", "25% cao nhất"]
    column_order = row_order + ["Chưa có assessment được chấm"]
    grouped = (
        frame.groupby(
            ["engagement_quartile", "assessment_score_quartile"],
            observed=True,
        )
        .agg(at_risk_rate=("At_Risk", "mean"), attempts=("id_student", "size"))
        .reset_index()
    )
    rate = grouped.pivot(
        index="engagement_quartile",
        columns="assessment_score_quartile",
        values="at_risk_rate",
    ).reindex(index=row_order, columns=column_order)
    count = (
        grouped.pivot(
            index="engagement_quartile",
            columns="assessment_score_quartile",
            values="attempts",
        )
        .reindex(index=row_order, columns=column_order)
        .fillna(0)
    )
    text = np.empty(rate.shape, dtype=object)
    for row in range(rate.shape[0]):
        for column in range(rate.shape[1]):
            value = rate.iloc[row, column]
            text[row, column] = (
                "—"
                if pd.isna(value)
                else f"{value:.0%}<br>N={int(count.iloc[row, column]):,}"
            )
    figure = go.Figure(
        go.Heatmap(
            z=rate.values,
            x=rate.columns,
            y=rate.index,
            text=text,
            texttemplate="%{text}",
            customdata=count.values,
            colorscale=RISK_SCALE,
            zmin=0,
            zmax=1,
            colorbar={"title": "Tỷ lệ trượt", "tickformat": ".0%"},
            hovertemplate=(
                "Mức tham gia trực tuyến=%{y}<br>Điểm bài tập=%{x}<br>Tỷ lệ trượt=%{z:.1%}"
                "<br>N=%{customdata:,}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="8 · Khi mức tham gia trực tuyến và điểm bài tập cùng thấp",
        xaxis_title="Nhóm điểm bài tập đến ngày 105",
        yaxis_title="Nhóm mức tham gia học trực tuyến đến ngày 105",
    )
    polish_figure(figure, height=550, legend="none")
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "25% thấp nhất/cao nhất được tính trong phạm vi phân tích. Mỗi ô kết hợp hai yếu tố; "
        "N là số lượt học trong ô."
    )
    low_rate = float(rate.loc["25% thấp nhất", "25% thấp nhất"])
    low_n = int(count.loc["25% thấp nhất", "25% thấp nhất"])
    high_rate = float(rate.loc["25% cao nhất", "25% cao nhất"])
    high_n = int(count.loc["25% cao nhất", "25% cao nhất"])
    return low_rate, low_n, high_rate, high_n


def render_model_kpis(frame: pd.DataFrame) -> tuple[float, float, float]:
    accuracy = frame["actual_at_risk"].eq(frame["predicted_at_risk"]).mean()
    actual_positive = frame["actual_at_risk"].eq(1)
    recall = (
        frame.loc[actual_positive, "predicted_at_risk"].eq(1).mean()
        if actual_positive.any()
        else float("nan")
    )
    predicted_positive = frame["predicted_at_risk"].eq(1)
    precision = (
        frame.loc[predicted_positive, "actual_at_risk"].eq(1).mean()
        if predicted_positive.any()
        else float("nan")
    )
    cards = st.columns(3)
    cards[0].metric("Tỷ lệ dự đoán đúng", format_percent(accuracy))
    cards[1].metric("Tỷ lệ phát hiện lượt trượt", format_percent(recall))
    cards[2].metric("Tỷ lệ cảnh báo trượt chính xác", format_percent(precision))
    return float(accuracy), float(recall), float(precision)


def render_interaction_heatmap(
    frame: pd.DataFrame,
) -> tuple[str, str, float, int]:
    grouped = (
        frame.groupby(["highest_education", "imd_band"], observed=True)
        .agg(
            at_risk_rate=("actual_at_risk", "mean"),
            attempts=("id_student", "size"),
        )
        .reset_index()
    )
    grouped["education_label"] = grouped["highest_education"].map(
        EDUCATION_LABELS
    ).fillna("Không xác định")
    education_order = [
        EDUCATION_LABELS.get(value, value)
        for value in [
            "No Formal quals",
            "Lower Than A Level",
            "A Level or Equivalent",
            "HE Qualification",
            "Post Graduate Qualification",
            "Không xác định",
        ]
        if value in set(grouped["highest_education"])
    ]
    imd_order = sorted(
        grouped["imd_band"].astype(str).unique(), key=imd_sort_key
    )
    rate = grouped.pivot(
        index="education_label", columns="imd_band", values="at_risk_rate"
    ).reindex(index=education_order, columns=imd_order)
    count = (
        grouped.pivot(
            index="education_label", columns="imd_band", values="attempts"
        )
        .reindex(index=education_order, columns=imd_order)
        .fillna(0)
    )
    text = np.empty(rate.shape, dtype=object)
    for row in range(rate.shape[0]):
        for column in range(rate.shape[1]):
            value = rate.iloc[row, column]
            text[row, column] = (
                "—"
                if pd.isna(value)
                else f"{value:.0%}<br>N={int(count.iloc[row, column]):,}"
            )
    figure = go.Figure(
        go.Heatmap(
            z=rate.values,
            x=rate.columns,
            y=rate.index,
            text=text,
            texttemplate="%{text}",
            customdata=count.values,
            colorscale=RISK_SCALE,
            zmin=0,
            zmax=1,
            colorbar={"title": "Tỷ lệ trượt", "tickformat": ".0%"},
            hovertemplate=(
                "Học vấn=%{y}<br>Mức khó khăn khu vực=%{x}<br>Tỷ lệ trượt=%{z:.1%}"
                "<br>N=%{customdata:,}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="9 · Trình độ đầu vào và mức khó khăn kinh tế của khu vực",
        xaxis_title="Nhóm mức khó khăn kinh tế–xã hội của khu vực",
        yaxis_title="Học vấn trước đó",
    )
    polish_figure(figure, height=535, legend="none")
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Mỗi ô kết hợp hai yếu tố; màu thể hiện tỷ lệ trượt, N là số lượt học. "
        "Biểu đồ chỉ mô tả bối cảnh, không dùng để quy kết hoàn cảnh gây ra kết quả."
    )
    eligible = grouped.loc[grouped["attempts"].ge(30)].sort_values(
        "at_risk_rate", ascending=False
    )
    top = (
        eligible.iloc[0]
        if not eligible.empty
        else grouped.sort_values("at_risk_rate", ascending=False).iloc[0]
    )
    return (
        str(top["highest_education"]),
        str(top["imd_band"]),
        float(top["at_risk_rate"]),
        int(top["attempts"]),
    )


def render_attempt_boxplot(frame: pd.DataFrame) -> tuple[float, float]:
    data = frame.dropna(subset=["assessment_weighted_score_cutoff"]).copy()
    data["previous_attempt_group"] = np.where(
        data["num_of_prev_attempts"].ge(3),
        "3+",
        data["num_of_prev_attempts"].astype(int).astype(str),
    )
    order = [
        value
        for value in ["0", "1", "2", "3+"]
        if value in set(data["previous_attempt_group"])
    ]
    figure = px.box(
        data,
        x="previous_attempt_group",
        y="assessment_weighted_score_cutoff",
        color="previous_attempt_group",
        category_orders={"previous_attempt_group": order},
        color_discrete_sequence=["#93C5FD", "#60A5FA", "#F59E0B", "#DC2626"],
        points="outliers",
        labels={
            "previous_attempt_group": "Số lần từng học học phần này",
            "assessment_weighted_score_cutoff": "Điểm bài tập có trọng số đến ngày 105",
        },
        title="10 · Điểm bài tập theo số lần từng học lại học phần",
    )
    figure.update_traces(
        hovertemplate="Nhóm=%{x}<br>Điểm=%{y:.1f}<extra></extra>"
    )
    polish_figure(figure, height=500, legend="none")
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Nhóm 3+ gộp các giá trị từ 3 đến 6 để giữ cỡ mẫu; điểm chỉ dùng dữ liệu có trước hoặc tại ngày 105."
    )
    medians = data.groupby("previous_attempt_group", observed=True)[
        "assessment_weighted_score_cutoff"
    ].median()
    return float(medians.get("0", np.nan)), float(medians.get("3+", np.nan))


def build_interaction_story(
    frame: pd.DataFrame, min_cell_n: int = 30
) -> list[str]:
    """Build filter-aware Page 3 insights without changing chart calculations."""
    insufficient = "Không đủ số lượng quan sát để đưa ra nhận xét."
    if frame.empty:
        return [insufficient]

    low_profile = frame.loc[
        frame["engagement_quartile"].astype(str).eq("25% thấp nhất")
        & frame["assessment_score_quartile"].eq("25% thấp nhất")
    ]
    high_profile = frame.loc[
        frame["engagement_quartile"].astype(str).eq("25% cao nhất")
        & frame["assessment_score_quartile"].eq("25% cao nhất")
    ]
    if len(low_profile) >= min_cell_n and len(high_profile) >= min_cell_n:
        low_rate = float(low_profile["At_Risk"].mean())
        high_rate = float(high_profile["At_Risk"].mean())
        profile_message = (
            "Tín hiệu phối hợp: trong phạm vi đang lọc, nhóm có tương tác VLE và "
            f"điểm assessment cùng thuộc 25% thấp nhất có tỷ lệ Fail {low_rate:.1%} "
            f"trên {len(low_profile):,} lượt học. Ở nhóm cùng thuộc 25% cao nhất, "
            f"tỷ lệ này là {high_rate:.1%} trên {len(high_profile):,} lượt học, tạo "
            f"khoảng cách {abs(low_rate - high_rate) * 100:.1f} điểm phần trăm. Đây là tín "
            "hiệu để ưu tiên rà soát tại ngày 105, không phải bằng chứng rằng hai yếu "
            "tố này trực tiếp gây ra Fail."
        )
    else:
        profile_message = insufficient

    first_time = frame.loc[frame["num_of_prev_attempts"].lt(1)]
    repeated = frame.loc[frame["num_of_prev_attempts"].ge(1)]
    if len(first_time) >= min_cell_n and len(repeated) >= min_cell_n:
        first_rate = float(first_time["At_Risk"].mean())
        repeat_rate = float(repeated["At_Risk"].mean())
        repeat_gap = repeat_rate - first_rate
        comparison = "cao hơn" if repeat_gap >= 0 else "thấp hơn"
        previous_message = (
            "Lịch sử học lại: các lượt học có ít nhất một lần học trước ghi nhận tỷ "
            f"lệ Fail {repeat_rate:.1%} (N={len(repeated):,}), {comparison} "
            f"{abs(repeat_gap) * 100:.1f} điểm phần trăm so với nhóm học lần đầu "
            f"({first_rate:.1%}, N={len(first_time):,}). Vì vậy, kinh nghiệm học trước "
            "không nên mặc nhiên được xem là yếu tố bảo vệ; cố vấn cần tìm hiểu trở "
            "ngại còn tồn tại từ lần học trước."
        )
    else:
        previous_message = insufficient

    context = (
        frame.groupby(["highest_education", "imd_band"], observed=True)
        .agg(fail_rate=("At_Risk", "mean"), attempts=("id_student", "size"))
        .reset_index()
    )
    eligible = context.loc[context["attempts"].ge(min_cell_n)].sort_values(
        ["fail_rate", "attempts"], ascending=[False, False]
    )
    if eligible.empty:
        context_message = insufficient
    else:
        top = eligible.iloc[0]
        education = EDUCATION_LABELS.get(
            str(top["highest_education"]), str(top["highest_education"])
        )
        context_message = (
            f"Bối cảnh học vấn và IMD: trong các tổ hợp có ít nhất {min_cell_n:,} "
            f"lượt học, nhóm {education} tại mức IMD {top['imd_band']} ghi nhận tỷ lệ "
            f"Fail cao nhất là {float(top['fail_rate']):.1%} (N={int(top['attempts']):,}). "
            "Các biến này chỉ nên dùng để nhận diện nhu cầu hỗ trợ ở cấp nhóm, không "
            "làm căn cứ duy nhất để đánh giá một cá nhân."
        )

    return [profile_message, previous_message, context_message]


def render_interaction_page() -> None:
    render_header(
        "Trang 3 · Kết hợp nhiều yếu tố",
        "Điều gì xảy ra khi nhiều yếu tố bất lợi xuất hiện cùng lúc?",
        "Xem đồng thời mức tham gia trực tuyến, điểm bài tập và lịch sử học lại.",
        "Dữ liệu có đến ngày 105 · mỗi ô đều ghi số lượt học (N)",
    )
    render_term_guide(
        [
            ("Kết hợp yếu tố", "xem hai đặc điểm cùng lúc thay vì tách riêng"),
            ("Nhóm 25%", "chia dữ liệu thành bốn nhóm có quy mô gần bằng nhau"),
            ("IMD", "nhóm mức khó khăn kinh tế–xã hội của khu vực cư trú"),
            ("N", "số lượt học trong nhóm đang hiển thị"),
        ]
    )
    base = prepare_interaction_frame()
    genders, age_bands, education_levels, imd_bands = context_filters(
        base, key_prefix="interaction"
    )
    frame = filter_snapshot_context(
        base, genders, age_bands, education_levels, imd_bands
    )
    if frame.empty:
        st.warning("Bộ lọc hiện tại không có lượt học trong snapshot ngày 105.")
        return

    render_story(
        "STORY · Nhận định khi kết hợp nhiều yếu tố",
        build_interaction_story(frame, min_cell_n=30),
    )
    render_engagement_assessment_heatmap(frame)
    render_interaction_heatmap(frame)
    render_attempt_boxplot(frame)


def render_probability_validation(frame: pd.DataFrame) -> tuple[float, float]:
    data = frame.copy()
    bin_count = min(10, max(2, data["risk_probability"].nunique()))
    labels = [f"Nhóm {value}" for value in range(1, bin_count + 1)]
    data["probability_group"] = pd.qcut(
        data["risk_probability"].rank(method="first"),
        q=bin_count,
        labels=labels,
    )
    grouped = (
        data.groupby("probability_group", observed=True)
        .agg(
            predicted_probability=("risk_probability", "mean"),
            actual_fail_rate=("actual_at_risk", "mean"),
            attempts=("actual_at_risk", "size"),
        )
        .reset_index()
    )
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=grouped["probability_group"],
            y=grouped["actual_fail_rate"],
            name="Tỷ lệ trượt thực tế",
            marker_color="#DC2626",
            customdata=grouped[["attempts"]],
            text=grouped["actual_fail_rate"],
            texttemplate="%{text:.0%}",
            textposition="outside",
            hovertemplate=(
                "%{x}<br>Trượt thực tế=%{y:.1%}<br>N=%{customdata[0]:,}<extra></extra>"
            ),
        )
    )
    figure.add_trace(
        go.Scatter(
            x=grouped["probability_group"],
            y=grouped["predicted_probability"],
            name="Xác suất model dự báo",
            mode="lines+markers",
            line={"color": "#2563EB", "width": 3},
            marker={"size": 9},
            hovertemplate="%{x}<br>Xác suất dự báo=%{y:.1%}<extra></extra>",
        )
    )
    figure.update_layout(
        title="11 · Nguy cơ dự báo cao hơn có đi cùng tỷ lệ trượt thực tế cao hơn?",
        xaxis_title="Từ 10% xác suất thấp nhất đến 10% cao nhất",
        yaxis_title="Tỷ lệ / xác suất trượt",
        barmode="overlay",
    )
    figure.update_yaxes(tickformat=".0%", range=[0, 1.08])
    polish_figure(figure, height=500)
    st.plotly_chart(figure, width="stretch")
    low_rate = float(grouped.iloc[0]["actual_fail_rate"])
    high_rate = float(grouped.iloc[-1]["actual_fail_rate"])
    st.caption(
        f"Nhóm xác suất thấp nhất có {low_rate:.1%} trượt thực tế; nhóm cao nhất là "
        f"{high_rate:.1%}. Cột đỏ là kết quả thật, đường xanh là xác suất model dự báo."
    )
    return low_rate, high_rate


def render_confusion_matrix(frame: pd.DataFrame) -> tuple[int, int, int, int]:
    counts = frame["error_type"].value_counts()
    tp, tn, fp, fn = (
        int(counts.get(value, 0)) for value in ["TP", "TN", "FP", "FN"]
    )
    matrix = np.array([[tn, fp], [fn, tp]])
    row_totals = matrix.sum(axis=1, keepdims=True)
    row_rates = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(matrix, dtype=float),
        where=row_totals != 0,
    )
    text = np.array(
        [
            [
                f"{matrix[row, column]:,}<br>{row_rates[row, column]:.1%}"
                for column in range(2)
            ]
            for row in range(2)
        ]
    )
    figure = go.Figure(
        go.Heatmap(
            z=row_rates,
            x=["Dự báo qua môn", "Dự báo trượt"],
            y=["Thực tế qua môn", "Thực tế trượt"],
            text=text,
            texttemplate="%{text}",
            customdata=matrix,
            colorscale=[[0, "#EFF6FF"], [1, "#1D4ED8"]],
            zmin=0,
            zmax=1,
            colorbar={"title": "Tỷ lệ trong<br>kết quả thật", "tickformat": ".0%"},
            hovertemplate=(
                "%{y}<br>%{x}<br>N=%{customdata:,}<br>Tỷ lệ=%{z:.1%}<extra></extra>"
            ),
        )
    )
    figure.update_layout(
        title="12 · Model dự báo đúng và sai ở đâu?",
        xaxis_title="Kết quả model dự báo",
        yaxis_title="Kết quả thực tế",
    )
    polish_figure(figure, height=500, legend="none")
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"Phát hiện đúng {tp:,} lượt trượt, bỏ sót {fn:,}; nhận diện đúng {tn:,} lượt "
        f"qua môn và cảnh báo nhầm {fp:,}. Phần trăm được tính trong từng hàng kết quả thật."
    )
    return tp, tn, fp, fn


def render_model_factors() -> None:
    """Explain the model with a small set of actionable global coefficients."""
    coefficients = model_table("model_coefficients.csv")
    required = {"feature", "coefficient", "odds_ratio", "direction"}
    missing = required.difference(coefficients.columns)
    if missing:
        raise ValueError(
            "model_coefficients.csv thiếu cột: " + ", ".join(sorted(missing))
        )
    friendly = {
        "vle_days_since_last_activity": "Lâu không hoạt động trên VLE",
        "assessment_missed_due_count": "Nhiều bài đến hạn chưa nộp",
        "num_of_prev_attempts": "Đã từng học học phần này",
        "vle_active_days_last_28_days": "Nhiều ngày học chủ động trong 28 ngày",
        "log1p_vle_resource_count_cutoff": "Sử dụng đa dạng tài nguyên VLE",
        "assessment_due_weighted_points_cutoff": "Tích lũy điểm ở bài đã đến hạn",
        "assessment_weighted_score_cutoff": "Điểm bài tập có trọng số cao",
    }
    data = coefficients.loc[coefficients["feature"].isin(friendly)].copy()
    data["Yếu tố"] = data["feature"].map(friendly)
    data["Liên hệ"] = np.where(
        data["coefficient"].gt(0), "Đi cùng nguy cơ cao hơn", "Đi cùng nguy cơ thấp hơn"
    )
    data = data.sort_values("coefficient")
    figure = px.bar(
        data,
        x="coefficient",
        y="Yếu tố",
        orientation="h",
        color="Liên hệ",
        color_discrete_map={
            "Đi cùng nguy cơ cao hơn": "#DC2626",
            "Đi cùng nguy cơ thấp hơn": "#0F766E",
        },
        custom_data=["odds_ratio"],
        labels={"coefficient": "Mức liên hệ trong mô hình (0 = không đổi)"},
        title="13 · Yếu tố nào đi cùng nguy cơ trượt cao hơn hoặc thấp hơn?",
    )
    figure.add_vline(x=0, line_color="#64748B", line_width=1)
    figure.update_traces(
        hovertemplate=(
            "%{y}<br>Hệ số=%{x:.3f}<br>Odds ratio=%{customdata[0]:.3f}<extra></extra>"
        )
    )
    polish_figure(figure, height=500)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Hệ số là kết quả toàn cục của mô hình, không thay đổi theo bộ lọc nhân khẩu học "
        "phía trên. Đây là mức liên hệ sau khi đã xét đồng thời các biến, không phải lời "
        "giải thích riêng cho từng lượt học hay bằng chứng nhân quả. Hệ số dương đi cùng "
        "nguy cơ trượt cao hơn; hệ số âm đi cùng nguy cơ thấp hơn."
    )


def build_model_operational_story(
    frame: pd.DataFrame, min_group_n: int = 30
) -> list[str]:
    """Build filter-aware operational insights from verified test predictions."""
    insufficient = "Không đủ số lượng quan sát để đưa ra nhận xét."
    if frame.empty or len(frame) < min_group_n:
        return [insufficient]

    counts = frame["error_type"].value_counts()
    tp = int(counts.get("TP", 0))
    tn = int(counts.get("TN", 0))
    fp = int(counts.get("FP", 0))
    fn = int(counts.get("FN", 0))
    actual_fail = tp + fn
    actual_non_fail = tn + fp
    threshold_values = frame["prediction_threshold"].dropna().unique()
    threshold = float(threshold_values[0]) if len(threshold_values) else float("nan")

    if (
        actual_fail >= min_group_n
        and actual_non_fail >= min_group_n
        and pd.notna(threshold)
    ):
        recall = tp / actual_fail
        coverage_message = (
            f"Độ bao phủ và khối lượng rà soát: tại ngưỡng phân loại {threshold:.3f}, "
            f"trong {actual_fail:,} lượt học thực sự Fail, mô hình phát hiện đúng "
            f"{tp:,} và bỏ sót {fn:,} lượt, tương ứng Recall {recall:.1%}. Trong "
            f"{actual_non_fail:,} lượt thực tế qua môn, mô hình tạo {fp:,} cảnh báo "
            "nhầm. Đây là sự đánh đổi tại ngưỡng hiện hành: giảm bỏ sót thường làm "
            "tăng khối lượng trường hợp mà cố vấn phải rà soát."
        )
        blind_spot_message = (
            f"Điểm mù: {fn:,} lượt học thực sự Fail không được mô hình cảnh báo. Vì "
            "vậy, kết quả mô hình chỉ nên dùng để sắp xếp ưu tiên hỗ trợ, không thay "
            "thế đánh giá chuyên môn hoặc quan sát trực tiếp của cố vấn."
        )
    else:
        coverage_message = insufficient
        blind_spot_message = insufficient

    high = frame.loc[frame["risk_level"].eq("High")]
    low = frame.loc[frame["risk_level"].eq("Low")]
    profile_columns = [
        "vle_active_days_last_28_days",
        "vle_days_since_last_activity",
        "assessment_completion_rate_cutoff",
    ]
    if (
        len(high) >= min_group_n
        and len(low) >= min_group_n
        and not high[profile_columns].median().isna().any()
        and not low[profile_columns].median().isna().any()
    ):
        high_median = high[profile_columns].median()
        low_median = low[profile_columns].median()
        profile_message = (
            "Điểm chạm có thể hành động: trong phạm vi đang lọc, nhóm dự báo rủi ro "
            f"cao có trung vị {high_median['vle_active_days_last_28_days']:.0f} ngày "
            f"hoạt động VLE trong 28 ngày gần nhất, cách lần tương tác gần nhất "
            f"{high_median['vle_days_since_last_activity']:.0f} ngày và hoàn thành "
            f"{high_median['assessment_completion_rate_cutoff']:.0%} assessment đến "
            f"hạn (N={len(high):,}). Nhóm rủi ro thấp lần lượt là "
            f"{low_median['vle_active_days_last_28_days']:.0f} ngày, "
            f"{low_median['vle_days_since_last_activity']:.0f} ngày và "
            f"{low_median['assessment_completion_rate_cutoff']:.0%} (N={len(low):,}). "
            "Đây là những tín hiệu phù hợp để mở đầu việc kiểm tra trở ngại học tập, "
            "không phải bằng chứng về tác động của một biện pháp can thiệp."
        )
    else:
        profile_message = insufficient

    return [coverage_message, blind_spot_message, profile_message]


def render_prediction_page() -> None:
    render_header(
        "Trang 4 · Mô hình dự báo",
        "Mô hình nhận diện nguy cơ trượt dựa trên yếu tố nào?",
        "Đánh giá độ tin cậy và xác định các yếu tố liên quan đến Fail ở mức tổng hợp.",
        "Tập kiểm tra độc lập · Withdrawn không tham gia huấn luyện · dashboard không huấn luyện lại model",
    )
    render_term_guide(
        [
            ("Logistic Regression", "mô hình ước lượng xác suất trượt học phần"),
            ("Tỷ lệ dự đoán đúng", "trong 100 lượt học, mô hình đoán đúng bao nhiêu lượt"),
            ("Recall trượt", "trong 100 lượt thực sự trượt, mô hình cảnh báo được bao nhiêu lượt"),
            ("Bỏ sót", "lượt thực sự trượt nhưng mô hình không cảnh báo"),
        ]
    )
    st.markdown(
        """
        <div class="definition-box"><b>Mô hình dự đoán gì?</b><br>
        Mô hình dùng thông tin đăng ký, tiến độ bài tập và hoạt động VLE có đến ngày 105
        để ước lượng xác suất kết quả cuối là <b>Fail</b>. Pass và Distinction là nhóm đối chứng;
        Withdrawn được loại khỏi model. Kết quả được tổng hợp để đánh giá độ tin cậy
        và xác định những yếu tố liên quan nổi bật.
        Đây là <b>cảnh báo giữa khóa</b>, không phải dự đoán điểm số chính xác.</div>
        """,
        unsafe_allow_html=True,
    )
    try:
        verification = model_table("model_verification.csv")
        if "status" not in verification or not verification["status"].eq("PASS").all():
            st.error("Kiểm tra mô hình chưa đạt; dashboard không công bố dự báo.")
            return
        base = prepare_risk_frame()
        metrics = model_table("model_metrics.csv")
        test_metric = metrics.loc[
            metrics["model_name"].eq("logistic_regression")
            & metrics["dataset_split"].eq("test")
        ].iloc[0]
        intervals = model_table("model_confidence_intervals.csv")
        accuracy_ci = intervals.loc[intervals["metric"].eq("accuracy")].iloc[0]
    except (FileNotFoundError, ValueError) as exc:
        st.error(str(exc))
        return

    genders, age_bands, education_levels, imd_bands = context_filters(
        base, key_prefix="model"
    )
    frame = filter_context(
        base, genders, age_bands, education_levels, imd_bands, []
    )
    if frame.empty:
        st.warning("Bộ lọc hiện tại không có lượt học trong tập dữ liệu kiểm tra.")
        return

    accuracy, recall, precision = render_model_kpis(frame)
    st.caption(
        f"Các chỉ số phía trên tính trên {len(frame):,} lượt học trong bộ lọc hiện tại. "
        f"Trên toàn bộ tập kiểm tra độc lập: Accuracy {test_metric['accuracy']:.1%} "
        f"(KTC 95% {accuracy_ci['lower_95']:.1%}–{accuracy_ci['upper_95']:.1%}), "
        f"Recall trượt {test_metric['recall_at_risk']:.1%}, Balanced Accuracy "
        f"{test_metric['balanced_accuracy']:.1%}, ROC-AUC {test_metric['roc_auc']:.3f}."
    )

    render_story(
        "STORY · Dự báo giúp hiểu yếu tố nào?",
        build_model_operational_story(frame),
    )

    st.subheader("Độ tin cậy và các yếu tố liên quan đến dự báo Fail")
    render_probability_validation(frame)
    render_confusion_matrix(frame)
    render_model_factors()
    st.info(
        "Định hướng hành động: trong số các hệ số toàn cục đang hiển thị, các tín hiệu "
        "hành vi đến ngày 105 có độ lớn nổi bật. Gián đoạn VLE và bài đến hạn chưa nộp "
        "là những điểm chạm để cố vấn ưu tiên rà soát; việc duy trì hoạt động, hoàn thành "
        "assessment và củng cố điểm quá trình là nội dung phù hợp để trao đổi hỗ trợ. "
        "Dashboard không ước lượng tác động nhân quả của một biện pháp can thiệp."
    )


def main() -> None:
    apply_dashboard_css()
    page = page_navigation()
    if page == "Outcome & Geography":
        render_overview_page()
    elif page == "Learning Behavior":
        render_behavior_page()
    elif page == "Interaction Analysis":
        render_interaction_page()
    else:
        render_prediction_page()


if __name__ == "__main__":
    main()
