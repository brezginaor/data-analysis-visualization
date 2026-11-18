"""Plotly-фигуры дашборда. Каждая функция: готовые данные -> фигура, без расчётов метрик."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from loader import FP, NO_VERDICT, OTHER, OUTCOMES, REAL, SEVERITY_ORDER, ru_duration, ru_int, ru_pct

FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#ecebe6"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

# Один цвет — один смысл на всех графиках: оранжевый — реальный инцидент,
# синий — ложное срабатывание, серые — исходы без пользы для анализа.
OUTCOME_COLORS = {REAL: "#eb6834", FP: "#2a78d6", OTHER: "#a9a79f", NO_VERDICT: "#dcdbd4"}

# Критичность на графике скорости: критическая выделена тёмной толстой линией,
# остальные уровни — серые от тёмного к светлому (порядок уровней сохраняется).
SEVERITY_LINES = {
    "Критическая": (INK, 3.0),
    "Высокая": ("#6f6d68", 2.0),
    "Средняя": ("#9a9891", 2.0),
    "Низкая": ("#c3c2b7", 2.0),
    "Отсутствует": ("#dcdbd4", 2.0),
}

# Доля нарушений SLA — последовательная шкала одного цвета (красного), от светлого к тёмному.
SLA_SCALE = [[0.0, "#fdf0ee"], [0.5, "#ec8a7f"], [1.0, "#a3211b"]]

MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]
PLOT_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}


LEGEND_SPACE = 60  # место над графиком под легенду в одну-две строки


def base(fig: go.Figure, left: int = 48, right: int = 16, top: int = LEGEND_SPACE, bottom: int = 40) -> go.Figure:
    fig.update_layout(
        template="none",
        font=dict(family=FONT, size=12, color=INK_2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=left, r=right, t=top, b=bottom),
        separators=", ",
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(family=FONT, size=12, color=INK)),
        # Легенда привязана к верхнему левому углу всей фигуры, а не к области графика:
        # при любой ширине экрана она остаётся в отведённой сверху полосе и не наезжает на данные.
        legend=dict(orientation="h", xref="container", yref="container", x=0, xanchor="left",
                    y=1, yanchor="top", font=dict(size=12),
                    itemclick=False, itemdoubleclick=False, traceorder="normal"),
        dragmode=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="", fixedrange=True,
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, gridwidth=1, zeroline=False, showline=False,
                     ticks="", fixedrange=True, tickfont=dict(color=MUTED))
    return fig


def empty(message: str = "Нет карточек для выбранных фильтров") -> go.Figure:
    fig = base(go.Figure())
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=13, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ---------------------------------------------------------------- 1. Поток
def _time_ticks(index: pd.DatetimeIndex, freq: str) -> tuple[list, list]:
    if freq == "D":
        step = max(1, int(np.ceil(len(index) / 10)))
        ticks = index[::step]
        return list(ticks), [d.strftime("%d.%m") for d in ticks]
    months = pd.date_range(index.min().to_period("M").to_timestamp(), index.max(), freq="MS")
    months = months[months >= index.min() - pd.Timedelta(6, unit="D")]
    step = max(1, int(np.ceil(len(months) / 9)))
    ticks = months[::step]
    labels, year = [], None
    for d in ticks:
        labels.append(MONTHS[d.month - 1] + (f"<br>{d.year}" if d.year != year else ""))
        year = d.year
    return list(ticks), labels


def flow(counts: pd.DataFrame, freq: str) -> go.Figure:
    if counts.empty or counts.to_numpy().sum() == 0:
        return empty()
    fig = base(go.Figure(), left=52)
    unit = "в день" if freq == "D" else "в неделю"
    for outcome in OUTCOMES:
        fig.add_trace(go.Bar(
            x=counts.index, y=counts[outcome], name=outcome,
            marker=dict(color=OUTCOME_COLORS[outcome], line=dict(color=SURFACE, width=0.5)),
            hovertemplate="%{y:,} — " + outcome + "<extra></extra>",
        ))
    ticks, labels = _time_ticks(counts.index, freq)
    fig.update_layout(barmode="stack", bargap=0.18, barcornerradius=2, hovermode="x unified")
    fig.update_xaxes(type="date", tickvals=ticks, ticktext=labels,
                     hoverformat="%d.%m.%Y" if freq == "D" else "неделя с %d.%m.%Y")
    fig.update_yaxes(title=dict(text=f"карточек {unit}", font=dict(color=MUTED, size=12)),
                     rangemode="tozero", tickformat=",d", nticks=5)
    return fig


# ---------------------------------------------------------------- 2. Правила
def _short(text: str, width: int = 40) -> str:
    return text if len(text) <= width else text[: width - 1] + "…"


def rules(table: pd.DataFrame) -> go.Figure:
    if table.empty:
        return empty()
    labels = [_short(rule) for rule in table.index]
    fig = base(go.Figure(), left=10, right=24, bottom=8)
    for outcome in OUTCOMES:
        fig.add_trace(go.Bar(
            x=table[outcome], y=labels, orientation="h", name=outcome,
            customdata=list(table.index),
            marker=dict(color=OUTCOME_COLORS[outcome], line=dict(color=SURFACE, width=1)),
            hovertemplate="<b>%{customdata}</b><br>" + outcome + ": %{x:,}<extra></extra>",
        ))
    for label, total, share in zip(labels, table["total"], table["share"]):
        fig.add_annotation(x=total, y=label, text=f"  {ru_int(total)} · {ru_pct(share)}",
                           xanchor="left", showarrow=False, font=dict(size=12, color=INK_2))
    fig.update_layout(barmode="stack", bargap=0.35, barcornerradius=2)
    # Запас справа под подпись «число · доля», чтобы она не обрезалась на узком экране.
    fig.update_xaxes(visible=False, range=[0, table["total"].max() * 1.6])
    fig.update_yaxes(showgrid=False, autorange="reversed", automargin=True,
                     tickfont=dict(color=INK_2, size=12))
    return fig


# ---------------------------------------------------------------- 3. Скорость реакции
DURATION_TICKS = [(60, "1 мин"), (900, "15 мин"), (3600, "1 ч"), (4 * 3600, "4 ч"),
                  (86400, "1 сут"), (7 * 86400, "7 сут"), (30 * 86400, "30 сут")]


def reaction(groups: dict[str, np.ndarray], grid: np.ndarray, ecdf) -> go.Figure:
    if not groups:
        return empty("Мало карточек с датой взятия в работу")
    fig = base(go.Figure(), left=52, right=24)
    labels = [ru_duration(x) for x in grid]
    for level in SEVERITY_ORDER:
        if level not in groups:
            continue
        values = groups[level]
        color, width = SEVERITY_LINES[level]
        fig.add_trace(go.Scatter(
            x=grid, y=ecdf(values, grid), mode="lines", name=level, line=dict(color=color, width=width, shape="hv"),
            customdata=labels,
            hovertemplate=level + ": за %{customdata} — %{y:.0%}<extra></extra>",
        ))
    # Медиана — место, где кривая пересекает 50 %. Подписываем только критическую.
    fig.add_hline(y=0.5, line=dict(color=AXIS, width=1))
    fig.add_annotation(x=0, xref="paper", y=0.5, text="половина карточек", showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(size=11, color=MUTED))
    if "Критическая" in groups:
        median = float(np.median(groups["Критическая"]))
        x = max(median, grid[0])
        # Подпись под линией 50 %: справа от точки, а у правого края — слева, чтобы не обрезалась.
        right_side = (np.log10(x) - np.log10(grid[0])) / (np.log10(grid[-1]) - np.log10(grid[0])) > 0.6
        fig.add_trace(go.Scatter(
            x=[x], y=[0.5], mode="markers+text", showlegend=False, hoverinfo="skip",
            marker=dict(size=10, color=INK, line=dict(color=SURFACE, width=2)),
            text=[f"медиана критических: {ru_duration(median)}"],
            textposition="bottom left" if right_side else "bottom right",
            textfont=dict(color=INK, size=12),
        ))
    ticks = [(v, t) for v, t in DURATION_TICKS if grid[0] <= v <= grid[-1] * 1.01]
    fig.update_xaxes(type="log", tickvals=[v for v, _ in ticks], ticktext=[t for _, t in ticks],
                     range=[np.log10(grid[0]), np.log10(grid[-1])], showgrid=True, gridcolor=GRID,
                     title=dict(text="время от создания карточки до взятия в работу", font=dict(color=MUTED)))
    fig.update_yaxes(range=[0, 1.02], tickformat=".0%", tickvals=[0, 0.25, 0.5, 0.75, 1],
                     title=dict(text="доля взятых в работу", font=dict(color=MUTED)))
    fig.update_layout(hovermode="x unified", margin=dict(b=52))
    return fig


# ---------------------------------------------------------------- 4. SLA
def sla(share: pd.DataFrame, counts: pd.DataFrame) -> go.Figure:
    if share.isna().all().all():
        return empty("Мало карточек с известным временем взятия в работу")
    top = float(np.nanmax(share.to_numpy()))
    zmax = max(0.1, np.ceil(top * 10) / 10)
    text = share.map(lambda v: "" if pd.isna(v) else ru_pct(v, 0))
    hover = [[("мало карточек" if pd.isna(share.iat[i, j]) else ru_pct(share.iat[i, j]))
              for j in range(share.shape[1])] for i in range(share.shape[0])]
    fig = base(go.Figure(), left=10, right=8, top=8, bottom=36)
    fig.add_trace(go.Heatmap(
        z=share.to_numpy(), x=list(share.columns), y=list(share.index), zmin=0, zmax=zmax,
        colorscale=SLA_SCALE, xgap=3, ygap=3, hoverongaps=True,
        customdata=np.dstack([np.array(hover, dtype=object), counts.to_numpy().astype(int)]),
        hovertemplate="<b>%{y}, %{x}</b><br>нарушен SLA: %{customdata[0]}"
                      "<br>карточек: %{customdata[1]:,}<extra></extra>",
        showscale=False,  # доля подписана в каждой ячейке — отдельная шкала не нужна
    ))
    for i, row in enumerate(share.index):
        for j, col in enumerate(share.columns):
            value = share.iat[i, j]
            if pd.isna(value):
                continue
            fig.add_annotation(x=col, y=row, text=text.iat[i, j], showarrow=False,
                               font=dict(size=12, color=SURFACE if value > zmax * 0.6 else INK))
    fig.update_xaxes(side="bottom", showgrid=False, tickangle=0, tickfont=dict(color=INK_2),
                     title=dict(text="время создания карточки, часы по Москве", font=dict(color=MUTED)))
    fig.update_yaxes(showgrid=False, autorange="reversed", automargin=True, tickfont=dict(color=INK_2))
    return fig
