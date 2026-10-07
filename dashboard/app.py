"""Four-page OULAD learning analytics dashboard.

The pages follow one analytical flow: outcomes and geography, learning
behaviour, multivariate interaction, then prediction and action. The app reads
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
    filter_attempts,
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
    "Prediction & Action": "4 · Dự đoán nguy cơ",
}
ATTEMPT_KEY = ["code_module", "code_presentation", "id_student"]
RESULT_ORDER = ["Distinction", "Pass", "Fail", "Withdrawn"]
RESULT_COLORS = {
    "Distinction": "#0F766E",
    "Pass": "#2563EB",
    "Fail": "#F97316",
    "Withdrawn": "#DC2626",
}
RISK_COLORS = {
    "Qua môn / Xuất sắc": "#2563EB",
    "Trượt": "#DC2626",
}
LEVEL_ORDER = ["High", "Medium", "Low"]
LEVEL_LABELS = {
    "High": "Cao · cần ưu tiên",
    "Medium": "Trung bình · cần theo dõi",
    "Low": "Thấp",
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


def filter_mart(
    frame: pd.DataFrame,
    modules: list[str],
    presentations: list[str],
    genders: list[str],
    regions: list[str],
) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for column, values in (
        ("code_module", modules),
        ("code_presentation", presentations),
        ("gender", genders),
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
            "- **Lượt học:** một lần sinh viên đăng ký học một học phần trong một đợt mở lớp.\n"
            "- **Bài tập/kiểm tra (assessment):** hoạt động được chấm điểm.\n"
            "- **Cảnh báo giữa khóa ngày 105:** chỉ dùng dữ liệu có sẵn đến ngày 105 để dự đoán nguy cơ trượt.\n"
            "- **AAA–GGG:** mã học phần đã được OULAD ẩn danh.\n"
            "- **B/J:** đợt học bắt đầu tháng 2/tháng 10.\n"
            "- Kết quả thể hiện liên hệ, không khẳng định nhân quả."
        )
    return page


def page_one_filters(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    st.subheader("Bộ lọc phân tích")
    columns = st.columns(3)
    modules = columns[0].multiselect(
        "Học phần ẩn danh · code_module",
        sorted(frame["code_module"].dropna().unique()),
        key="academic_modules",
        placeholder="Tất cả học phần",
    )
    available = frame.loc[frame["code_module"].isin(modules)] if modules else frame
    presentations = columns[1].multiselect(
        "Đợt mở lớp · code_presentation",
        sorted(available["code_presentation"].dropna().unique()),
        key="academic_presentations",
        placeholder="Tất cả đợt mở",
    )
    genders = columns[2].multiselect(
        "Giới tính · gender",
        sorted(frame["gender"].dropna().unique()),
        key="academic_genders",
        placeholder="Tất cả giới tính",
    )
    return modules, presentations, genders


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
    figure = px.choropleth(
        stats,
        geojson=geojson,
        locations="region",
        featureidkey="properties.region",
        color="at_risk_rate",
        color_continuous_scale=RISK_SCALE,
        range_color=(0, 1),
        custom_data=["region", "attempts", "at_risk_count", "at_risk_rate"],
        title="1 · Tỷ lệ trượt theo vùng cư trú",
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
        "Bấm một vùng để lọc KPI và biểu đồ cơ cấu kết quả trên Trang 1. "
        "Màu đậm hơn = tỷ lệ trượt cao hơn; màu không phản ánh số sinh viên. "
        "Bản đồ cho biết nơi cần xem xét sâu hơn, không tự chứng minh nơi cư trú gây ra kết quả."
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
    counts["final_result"] = pd.Categorical(
        counts["final_result"], RESULT_ORDER, ordered=True
    )
    return counts.sort_values([group_column, "final_result"])


def render_outcome_drill(frame: pd.DataFrame) -> None:
    valid_modules = set(frame["code_module"].astype(str))
    selected_module = st.session_state.get("outcome_drill_module")
    if selected_module not in valid_modules:
        selected_module = None
        st.session_state["outcome_drill_module"] = None

    if selected_module:
        top = st.columns([4, 1])
        top[0].markdown(
            f"**Drill-down:** Tất cả học phần → `{selected_module}` → Đợt mở lớp"
        )
        if top[1].button(
            "↑ Quay lại học phần", key="reset_outcome_drill", width="stretch"
        ):
            st.session_state["outcome_drill_module"] = None
            st.session_state["outcome_key_version"] = (
                st.session_state.get("outcome_key_version", 0) + 1
            )
            st.rerun()
        chart_frame = frame.loc[frame["code_module"].eq(selected_module)]
        group_column = "code_presentation"
        title = f"2 · Cơ cấu kết quả theo đợt mở lớp của {selected_module}"
        y_label = "Đợt mở lớp"
    else:
        st.caption(
            "AAA–GGG là mã học phần đã được ẩn danh. Bấm một thanh để xem các đợt mở lớp bên trong."
        )
        chart_frame = frame
        group_column = "code_module"
        title = "2 · Cơ cấu kết quả theo học phần ẩn danh"
        y_label = "Học phần ẩn danh"

    data = outcome_percentages(chart_frame, group_column)
    risk_order = (
        chart_frame.groupby(group_column, observed=True)["At_Risk"]
        .mean()
        .sort_values()
    )
    data["inside_label"] = data["share"].map(
        lambda value: f"{value:.0%}" if value >= .06 else ""
    )
    figure = px.bar(
        data,
        x="share",
        y=group_column,
        color="final_result",
        orientation="h",
        barmode="stack",
        category_orders={"final_result": RESULT_ORDER},
        color_discrete_map=RESULT_COLORS,
        custom_data=["attempts"],
        text="inside_label",
        labels={
            "share": "Tỷ trọng kết quả",
            group_column: y_label,
            "final_result": "Kết quả",
        },
        title=title,
    )
    figure.update_traces(
        texttemplate="%{text}",
        textposition="inside",
        hovertemplate=(
            f"{y_label}=%{{y}}<br>Kết quả=%{{fullData.name}}"
            "<br>Tỷ trọng=%{x:.1%}<br>N=%{customdata[0]:,}<extra></extra>"
        ),
    )
    figure.update_xaxes(tickformat=".0%", range=[0, 1])
    figure.update_yaxes(
        categoryorder="array", categoryarray=risk_order.index.tolist()
    )
    polish_figure(figure, height=470)
    for group, rate in risk_order.items():
        figure.add_annotation(
            x=.995,
            y=group,
            xref="x",
            yref="y",
            text=f"Trượt {rate:.0%}",
            showarrow=False,
            xanchor="right",
            bgcolor="rgba(255,255,255,.92)",
            bordercolor="#CBD5E1",
            borderpad=3,
            font={"size": 11, "color": INK},
        )
    event = st.plotly_chart(
        figure,
        width="stretch",
        on_select="rerun" if not selected_module else "ignore",
        selection_mode="points",
        key=(
            f"outcome_drill_{selected_module or 'module'}_"
            f"{st.session_state.get('outcome_key_version', 0)}"
        ),
    )
    if not selected_module:
        points = chart_points(event)
        if points:
            module = point_value(points[0], "y")
            if module in valid_modules:
                st.session_state["outcome_drill_module"] = str(module)
                st.rerun()
    st.caption(
        "Mỗi thanh luôn bằng 100%. Học phần có tỷ lệ trượt cao hơn nằm phía dưới; "
        "Fail và Withdrawn được hiển thị thành hai phần riêng. B = bắt đầu tháng 2, J = bắt đầu tháng 10."
    )


def render_vle_timeline(
    frame: pd.DataFrame,
    modules: list[str],
    presentations: list[str],
    genders: list[str],
    regions: list[str],
) -> tuple[float, float]:
    daily = dashboard_mart(
        "vle_daily_profile.csv.gz",
        (
            "code_module",
            "code_presentation",
            "gender",
            "region",
            "At_Risk",
            "date",
            "sum_click",
        ),
    )
    daily = filter_mart(daily, modules, presentations, genders, regions)
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
    if modules:
        deadlines = deadlines.loc[deadlines["code_module"].isin(modules)]
    if presentations:
        deadlines = deadlines.loc[
            deadlines["code_presentation"].isin(presentations)
        ]
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
    modules: list[str],
    presentations: list[str],
    genders: list[str],
    regions: list[str],
) -> float:
    submissions = dashboard_mart(
        "assessment_submissions.csv.gz",
        (
            "code_module",
            "code_presentation",
            "id_student",
            "date_submitted",
            "gender",
            "region",
            "num_of_prev_attempts",
            "At_Risk",
            "submission_delay",
            "score",
        ),
    )
    submissions = filter_mart(
        submissions, modules, presentations, genders, regions
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
        custom = np.column_stack(
            [
                group["id_student"],
                group["num_of_prev_attempts"],
                group["code_module"],
                group["code_presentation"],
            ]
        )
        figure.add_trace(
            go.Scattergl(
                x=group["submission_delay"],
                y=group["score"],
                mode="markers",
                name=label,
                customdata=custom,
                marker={
                    "size": 7 + 2 * group["num_of_prev_attempts"].clip(upper=6),
                    "color": RISK_COLORS[label],
                    "opacity": .42,
                    "line": {"color": "#FFFFFF", "width": .4},
                },
                hovertemplate=(
                    "Student=%{customdata[0]}<br>Module=%{customdata[2]}-%{customdata[3]}"
                    "<br>Trễ=%{x:.0f} ngày<br>Điểm=%{y:.1f}"
                    "<br>Lần học trước=%{customdata[1]:.0f}<extra></extra>"
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


def render_activity_treemap(
    modules: list[str],
    presentations: list[str],
    genders: list[str],
    regions: list[str],
) -> tuple[str, float]:
    activity = dashboard_mart(
        "vle_activity_summary.csv.gz",
        (
            "code_module",
            "code_presentation",
            "gender",
            "region",
            "activity_type",
            "sum_click",
        ),
    )
    activity = filter_mart(activity, modules, presentations, genders, regions)
    activity = activity.groupby("activity_type", as_index=False)["sum_click"].sum()
    if activity.empty:
        st.info("Không có click VLE trong phạm vi lọc.")
        return "—", float("nan")
    total = activity["sum_click"].sum()
    activity["share"] = activity["sum_click"] / total
    figure = px.treemap(
        activity,
        path=[px.Constant("Tất cả tài nguyên VLE"), "activity_type"],
        values="sum_click",
        color="sum_click",
        color_continuous_scale=["#DBEAFE", "#2563EB", "#1E3A8A"],
        custom_data=["sum_click", "share"],
        title="7 · Loại tài nguyên học trực tuyến nào được sử dụng nhiều nhất?",
    )
    figure.update_traces(
        marker={"line": {"color": "#FFFFFF", "width": 1.5}},
        texttemplate="<b>%{label}</b><br>%{customdata[1]:.1%}",
        hovertemplate=(
            "Loại=%{label}<br>Clicks=%{customdata[0]:,}"
            "<br>Tỷ trọng=%{customdata[1]:.1%}<extra></extra>"
        ),
    )
    figure.update_layout(coloraxis_showscale=False)
    polish_figure(figure, height=520, legend="none")
    st.plotly_chart(figure, width="stretch")
    st.caption(
        "Diện tích và độ đậm đều biểu diễn tổng lượt click; tỷ trọng tính trong phạm vi lọc hiện tại."
    )
    top = activity.sort_values("sum_click", ascending=False).iloc[0]
    return str(top["activity_type"]), float(top["share"])


def comparison_message(
    frame: pd.DataFrame,
    group_column: str,
    label: str,
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
        return (
            f"{label} {grouped.index[0]} có tỷ lệ trượt {row['mean']:.1%} "
            f"(N={int(row['size']):,})."
        )
    low, high = grouped.iloc[0], grouped.iloc[-1]
    return (
        f"Tỷ lệ trượt theo {label.lower()} cao nhất ở {grouped.index[-1]} "
        f"({high['mean']:.1%}, N={int(high['size']):,}) và thấp nhất ở "
        f"{grouped.index[0]} ({low['mean']:.1%}, N={int(low['size']):,})."
    )


def render_score_distribution(frame: pd.DataFrame) -> None:
    """Show the academic score distribution without hiding missing scores."""
    data = frame.dropna(subset=["assessment_score_mean_all_time"]).copy()
    if data.empty:
        st.info("Không có điểm bài tập/kiểm tra trong phạm vi lọc.")
        return
    figure = px.histogram(
        data,
        x="assessment_score_mean_all_time",
        color="final_result",
        category_orders={"final_result": RESULT_ORDER},
        color_discrete_map=RESULT_COLORS,
        nbins=20,
        barmode="overlay",
        opacity=.68,
        labels={
            "assessment_score_mean_all_time": "Điểm bài tập/kiểm tra trung bình",
            "count": "Số lượt học",
            "final_result": "Kết quả cuối",
        },
        title="3 · Phân bố điểm quá trình theo kết quả cuối",
    )
    figure.update_traces(
        hovertemplate="Khoảng điểm=%{x}<br>Số lượt=%{y:,}<extra></extra>"
    )
    figure.update_xaxes(range=[0, 100])
    polish_figure(figure, height=470)
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"Có {len(data):,}/{len(frame):,} lượt học có ít nhất một điểm được chấm. "
        "Biểu đồ bổ sung thước đo điểm quá trình bên cạnh kết quả cuối Pass/Fail/Withdrawn."
    )


def render_overview_page() -> None:
    render_header(
        "Trang 1 · Bức tranh kết quả học tập",
        "Sinh viên đang đạt kết quả như thế nào?",
        "Nhìn tổng thể kết quả, sau đó so sánh giữa học phần và khu vực cư trú.",
        "Phạm vi mô tả toàn khóa · một dòng = một lượt học của sinh viên trong một học phần–đợt mở",
    )
    render_term_guide(
        [
            ("AAA–GGG", "mã học phần đã được OULAD ẩn danh, không phải tên môn thật"),
            ("B/J", "đợt học bắt đầu tháng 2/tháng 10"),
            ("Trượt học phần", "kết quả cuối là Fail; rút học được tách riêng"),
            ("Lượt học", "một sinh viên trong một học phần–đợt mở"),
        ]
    )
    frame = analysis_data()
    modules, presentations, genders = page_one_filters(frame)
    base = filter_attempts(
        frame, modules=modules, presentations=presentations, genders=genders
    )
    if base.empty:
        st.warning("Bộ lọc hiện tại không có lượt học.")
        return

    active_region = st.session_state.get("academic_region")
    valid_regions = set(base["region"].dropna().astype(str))
    if active_region not in valid_regions:
        active_region = None
        st.session_state["academic_region"] = None
    effective = filter_attempts(
        base, regions=[active_region] if active_region else None
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
            comparison_message(effective, "code_module", "Học phần ẩn danh"),
            comparison_message(base, "region", "Vùng")
            + " Học phần và khu vực chỉ cho biết bối cảnh có chênh lệch, không chứng minh đó là nguyên nhân.",
        ],
    )

    st.subheader("Kết quả khác nhau ở đâu?")
    render_region_map(base, active_region)
    render_outcome_drill(effective)
    render_score_distribution(effective)


def filter_snapshot_context(
    frame: pd.DataFrame,
    modules: list[str],
    presentations: list[str],
    genders: list[str],
) -> pd.DataFrame:
    mask = pd.Series(True, index=frame.index)
    for column, values in (
        ("code_module", modules),
        ("code_presentation", presentations),
        ("gender", genders),
    ):
        if values:
            mask &= frame[column].isin(values)
    return frame.loc[mask].copy()


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
    modules, presentations, genders = page_one_filters(frame)
    filtered = filter_attempts(
        frame, modules=modules, presentations=presentations, genders=genders
    )
    snapshot = filter_snapshot_context(
        feature_snapshot(), modules, presentations, genders
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

    render_vle_timeline(filtered, modules, presentations, genders, [])
    render_completion_chart(snapshot)
    correlation = render_submission_scatter(modules, presentations, genders, [])
    top_activity, top_share = render_activity_treemap(
        modules, presentations, genders, []
    )
    delay_message = (
        "nộp càng trễ thường đi cùng điểm thấp hơn"
        if pd.notna(correlation) and correlation < 0
        else "chưa thấy xu hướng rõ giữa thời điểm nộp và điểm"
    )
    st.info(
        f"Đọc thêm: {delay_message}; loại tài nguyên được dùng nhiều nhất là "
        f"{top_activity} ({top_share:.1%} tổng click trong phạm vi lọc)."
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


def imd_sort_key(value: str) -> tuple[int, str]:
    if value == "Không xác định":
        return (999, value)
    try:
        return (int(value.split("-")[0].replace("%", "")), value)
    except ValueError:
        return (998, value)


def page_two_filters(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    st.subheader("Bộ lọc danh sách hỗ trợ")
    columns = st.columns(2)
    levels = columns[0].multiselect(
        "Mức nguy cơ dự đoán",
        LEVEL_ORDER,
        format_func=lambda value: LEVEL_LABELS[value],
        key="risk_levels",
        placeholder="Tất cả mức rủi ro",
    )
    imd_options = sorted(
        frame["imd_band"].astype(str).unique(), key=imd_sort_key
    )
    imd_bands = columns[1].multiselect(
        "Mức khó khăn kinh tế–xã hội của khu vực",
        imd_options,
        key="risk_imd_bands",
        placeholder="Tất cả nhóm khu vực",
    )
    return levels, imd_bands


def render_model_kpis(frame: pd.DataFrame) -> tuple[float, float, int]:
    accuracy = frame["actual_at_risk"].eq(frame["predicted_at_risk"]).mean()
    actual_positive = frame["actual_at_risk"].eq(1)
    recall = (
        frame.loc[actual_positive, "predicted_at_risk"].eq(1).mean()
        if actual_positive.any()
        else float("nan")
    )
    high_count = int(frame["risk_level"].eq("High").sum())
    cards = st.columns(3)
    cards[0].metric("Tỷ lệ dự đoán đúng", format_percent(accuracy))
    cards[1].metric("Tỷ lệ phát hiện lượt trượt", format_percent(recall))
    cards[2].metric("Lượt cần ưu tiên hỗ trợ", f"{high_count:,}")
    return float(accuracy), float(recall), high_count


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
    education_order = [
        value
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
        index="highest_education", columns="imd_band", values="at_risk_rate"
    ).reindex(index=education_order, columns=imd_order)
    count = (
        grouped.pivot(
            index="highest_education", columns="imd_band", values="attempts"
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
    modules, presentations, genders = page_one_filters(base)
    frame = filter_snapshot_context(base, modules, presentations, genders)
    if frame.empty:
        st.warning("Bộ lọc hiện tại không có lượt học trong snapshot ngày 105.")
        return

    low_profile = frame.loc[
        frame["engagement_quartile"].astype(str).eq("25% thấp nhất")
        & frame["assessment_score_quartile"].eq("25% thấp nhất")
    ]
    high_profile = frame.loc[
        frame["engagement_quartile"].astype(str).eq("25% cao nhất")
        & frame["assessment_score_quartile"].eq("25% cao nhất")
    ]
    low_rate = float(low_profile["At_Risk"].mean())
    high_rate = float(high_profile["At_Risk"].mean())
    low_n = len(low_profile)
    high_n = len(high_profile)
    previous_rates = frame.assign(
        has_previous_attempt=frame["num_of_prev_attempts"].ge(1)
    ).groupby("has_previous_attempt")["At_Risk"].mean()
    first_rate = float(previous_rates.get(False, np.nan))
    repeat_rate = float(previous_rates.get(True, np.nan))
    previous_message = (
        f"Sinh viên từng học học phần này trước đó có tỷ lệ trượt {repeat_rate:.1%}, "
        f"cao hơn nhóm học lần đầu {first_rate:.1%}."
        if pd.notna(first_rate) and pd.notna(repeat_rate)
        else "Phạm vi lọc chưa đủ dữ liệu để so sánh nhóm học lần đầu và nhóm từng học trước."
    )
    profile_message = (
        f"Khi mức tham gia trực tuyến và điểm bài tập đều thấp, tỷ lệ trượt là {low_rate:.1%}; "
        f"khi cả hai đều cao, tỷ lệ này chỉ còn {high_rate:.1%}."
        if pd.notna(low_rate) and pd.notna(high_rate) and low_n and high_n
        else "Phạm vi lọc chưa đủ cả hai hồ sơ VLE thấp–điểm thấp và VLE cao–điểm cao để so sánh."
    )
    render_story(
        "STORY · Nhận định khi kết hợp nhiều yếu tố",
        [
            profile_message,
            previous_message,
            "Kết luận trọng tâm là tiến độ học và mức tham gia; học vấn, hoàn cảnh khu vực và nơi ở chỉ là bối cảnh để xem xét thêm, không phải căn cứ quy kết cá nhân.",
        ],
    )
    render_engagement_assessment_heatmap(frame)
    render_interaction_heatmap(frame)
    render_attempt_boxplot(frame)


def render_risk_gauge(frame: pd.DataFrame) -> float:
    mean_probability = float(frame["risk_probability"].mean())
    threshold = float(frame["prediction_threshold"].iloc[0])
    medium_floor = threshold / 2
    figure = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=mean_probability * 100,
            number={"suffix": "%", "valueformat": ".1f", "font": {"size": 44}},
            title={"text": "Xác suất trượt trung bình", "font": {"size": 18}},
            gauge={
                "axis": {
                    "range": [0, 100],
                    "tickvals": [0, 20, 40, 60, 80, 100],
                    "ticktext": ["0%", "20%", "40%", "60%", "80%", "100%"],
                },
                "bar": {"color": INK, "thickness": .28},
                "steps": [
                    {"range": [0, medium_floor * 100], "color": "#BBF7D0"},
                    {"range": [medium_floor * 100, threshold * 100], "color": "#FDE68A"},
                    {"range": [threshold * 100, 100], "color": "#FCA5A5"},
                ],
                "threshold": {
                    "line": {"color": "#7C3AED", "width": 4},
                    "thickness": .8,
                    "value": threshold * 100,
                },
            },
        )
    )
    figure.update_layout(title="11 · Xác suất trượt của nhóm đang xem")
    polish_figure(figure, height=430, legend="none")
    figure.update_layout(margin={"l": 50, "r": 60, "t": 78, "b": 45})
    st.plotly_chart(figure, width="stretch")
    st.caption(
        f"Mức ưu tiên: thấp dưới {medium_floor:.1%}, trung bình từ {medium_floor:.1%} "
        f"đến dưới {threshold:.1%}, cao từ {threshold:.1%}. Vạch tím là ngưỡng phân loại "
        "đã chọn trên tập validation, không phải điểm đạt/trượt của môn học."
    )
    return mean_probability


def render_error_donut(frame: pd.DataFrame) -> tuple[int, int, int, int]:
    counts = frame["error_type"].value_counts()
    order = ["TP", "TN", "FP", "FN"]
    labels = {
        "TP": "Cảnh báo đúng lượt trượt",
        "TN": "Nhận diện đúng lượt qua môn",
        "FP": "Cảnh báo nhầm",
        "FN": "Bỏ sót lượt trượt",
    }
    colors = {
        "TP": "#DC2626",
        "TN": "#2563EB",
        "FP": "#F59E0B",
        "FN": "#7F1D1D",
    }
    data = pd.DataFrame(
        {"error_type": order, "count": [int(counts.get(value, 0)) for value in order]}
    )
    data["label"] = data["error_type"].map(labels)
    figure = px.pie(
        data,
        values="count",
        names="label",
        hole=.56,
        color="error_type",
        color_discrete_map=colors,
        category_orders={"error_type": order},
        title="12 · Đối chiếu dự đoán với kết quả thật",
    )
    figure.update_traces(
        textposition="inside",
        texttemplate="%{percent:.0%}",
        hovertemplate=(
            "%{label}<br>N=%{value:,}<br>Tỷ trọng=%{percent:.1%}<extra></extra>"
        ),
        marker={"line": {"color": "#FFFFFF", "width": 2}},
    )
    polish_figure(figure, height=500)
    figure.update_layout(
        legend={
            "orientation": "h",
            "yanchor": "top",
            "y": -.08,
            "xanchor": "left",
            "x": 0,
            "title_text": "",
        },
        margin={"l": 40, "r": 35, "t": 78, "b": 110},
    )
    st.plotly_chart(figure, width="stretch")
    tp, tn, fp, fn = (int(counts.get(value, 0)) for value in order)
    st.caption(
        f"Cảnh báo đúng lượt trượt: {tp:,} · nhận diện đúng lượt qua môn: {tn:,} · "
        f"cảnh báo nhầm: {fp:,} · bỏ sót: {fn:,}."
    )
    return tp, tn, fp, fn


def render_action_table(frame: pd.DataFrame) -> None:
    title, action = st.columns([4, 1])
    title.subheader("Danh sách sinh viên cần xem xét hỗ trợ")

    def select_high_risk() -> None:
        st.session_state["risk_levels"] = ["High"]

    action.button(
        "Chỉ xem mức nguy cơ cao",
        key="select_high_risk_action",
        on_click=select_high_risk,
        width="stretch",
        help="Một click để lọc toàn bộ Trang 4 về nhóm đã vượt ngưỡng dự báo trượt.",
    )
    candidates = frame.loc[frame["risk_level"].eq("High")].sort_values(
        "risk_probability", ascending=False
    )
    if candidates.empty:
        st.info("Bộ lọc hiện tại không có lượt học ở mức cảnh báo cao.")
        return
    table = candidates[
        [
            "id_student",
            "code_module",
            "code_presentation",
            "imd_band",
            "risk_probability",
            "predicted_status",
        ]
    ].head(100).rename(
        columns={
            "id_student": "Mã sinh viên",
            "code_module": "Học phần",
            "code_presentation": "Đợt mở lớp",
            "imd_band": "Nhóm khu vực",
            "predicted_status": "Kết quả dự đoán",
        }
    )
    table["Cảnh báo"] = table["risk_probability"].map(
        lambda value: "🟥" * max(1, int(round(value * 10)))
    )
    st.dataframe(
        table,
        width="stretch",
        hide_index=True,
        height=420,
        column_config={
            "risk_probability": st.column_config.NumberColumn(
                "Xác suất trượt", format="percent"
            ),
            "Cảnh báo": st.column_config.TextColumn(
                "Mức cảnh báo",
                help="Mỗi ô đỏ tương ứng khoảng 10 điểm phần trăm rủi ro.",
            ),
        },
    )
    st.caption(
        f"Hiển thị tối đa 100/{len(candidates):,} lượt có cảnh báo cao trong phạm vi bộ lọc. "
        "Danh sách giúp giảng viên biết nên kiểm tra và liên hệ ai trước."
    )


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
        labels={"coefficient": "Hệ số Logistic Regression"},
        title="13 · Những tín hiệu mô hình dùng để nhận diện nguy cơ trượt",
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
        "Đây là mức liên hệ chung của mô hình sau khi đã xét đồng thời các biến, không phải "
        "lời giải thích riêng cho từng sinh viên. Hệ số dương đi cùng nguy cơ trượt cao hơn; "
        "hệ số âm đi cùng nguy cơ thấp hơn."
    )


def risk_profile_message(frame: pd.DataFrame) -> str:
    high = frame.loc[frame["risk_level"].eq("High")]
    low = frame.loc[frame["risk_level"].eq("Low")]
    if high.empty or low.empty:
        return "Bộ lọc hiện tại chưa đủ cả nhóm cảnh báo cao và thấp để đối chiếu đặc điểm."
    high_active = high["vle_active_days_last_28_days"].median()
    low_active = low["vle_active_days_last_28_days"].median()
    high_gap = high["vle_days_since_last_activity"].median()
    low_gap = low["vle_days_since_last_activity"].median()
    high_completion = high["assessment_completion_rate_cutoff"].median()
    low_completion = low["assessment_completion_rate_cutoff"].median()
    return (
        f"Nhóm cảnh báo cao có trung vị {high_active:.0f} ngày hoạt động VLE trong 28 ngày "
        f"gần nhất và hoàn thành {high_completion:.0%} bài đã đến hạn; nhóm cảnh báo thấp lần lượt "
        f"là {low_active:.0f} ngày và {low_completion:.0%}. Khoảng cách từ lần hoạt động gần nhất "
        f"là {high_gap:.0f} so với {low_gap:.0f} ngày."
    )


def render_prediction_page() -> None:
    render_header(
        "Trang 4 · Dự đoán nguy cơ",
        "Cảnh báo giữa khóa: lượt học nào có nguy cơ trượt?",
        "Dùng dữ liệu đến ngày 105 để ước lượng xác suất kết quả cuối là Fail.",
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
        Với mỗi sinh viên trong một học phần–đợt mở, mô hình dùng thông tin đăng ký,
        tiến độ bài tập và hoạt động VLE có đến ngày 105 để ước lượng xác suất kết quả cuối
        là <b>Fail</b>. Pass và Distinction là nhóm đối chứng; Withdrawn được loại khỏi model.
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

    levels, imd_bands = page_two_filters(base)
    mask = pd.Series(True, index=base.index)
    if levels:
        mask &= base["risk_level"].isin(levels)
    if imd_bands:
        mask &= base["imd_band"].isin(imd_bands)
    frame = base.loc[mask].copy()
    if frame.empty:
        st.warning("Bộ lọc hiện tại không có lượt học trong tập dữ liệu kiểm tra.")
        return

    accuracy, recall, high_count = render_model_kpis(frame)
    st.caption(
        f"Các chỉ số phía trên tính trên {len(frame):,} lượt học trong bộ lọc hiện tại. "
        f"Trên toàn bộ tập kiểm tra độc lập: Accuracy {test_metric['accuracy']:.1%} "
        f"(KTC 95% {accuracy_ci['lower_95']:.1%}–{accuracy_ci['upper_95']:.1%}), "
        f"Recall trượt {test_metric['recall_at_risk']:.1%}, Balanced Accuracy "
        f"{test_metric['balanced_accuracy']:.1%}, ROC-AUC {test_metric['roc_auc']:.3f}."
    )

    fn = int(frame["error_type"].eq("FN").sum())
    render_story(
        "STORY · Mô hình dự đoán điều gì?",
        [
            f"Trong phạm vi đang xem, mô hình phân loại đúng khoảng {accuracy * 100:.0f}/100 lượt học và phát hiện khoảng {recall * 100:.0f}/100 lượt thực sự trượt.",
            f"Có {high_count:,} lượt vượt ngưỡng cảnh báo trượt; vẫn còn {fn:,} lượt trượt bị bỏ sót, vì vậy không được dùng model như quyết định tự động.",
            risk_profile_message(frame),
            "Hành động phù hợp là kiểm tra bài đến hạn, nhắc quay lại VLE và hỗ trợ nội dung còn yếu; không dùng hoàn cảnh cá nhân để quy kết.",
        ],
    )

    st.subheader("Kết quả dự đoán và danh sách ưu tiên hỗ trợ")
    left, right = st.columns(2)
    with left:
        render_risk_gauge(frame)
    with right:
        render_error_donut(frame)
    render_model_factors()
    render_action_table(frame)
    st.info(
        "Danh sách là công cụ ưu tiên hỗ trợ; không dùng model để tự động quyết định kết quả hay xử phạt người học."
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
