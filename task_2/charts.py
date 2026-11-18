"""Plotly-фигуры дашборда задания 2. Каждая функция: готовые данные -> фигура."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from loader import MONTH_NAMES, THRESHOLD, month_label, ru_int, ru_pct

FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#ecebe6"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

# Один цвет — один смысл, как в задании 1: оранжевый — реальный инцидент.
# Тёмно-красный — только пропущенные инциденты (главная метрика задания).
REAL = "#eb6834"
MISSED = "#b42318"
NEUTRAL = "#6f6d68"
LEGEND_SPACE = 60
PLOT_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}
SHORT_MONTHS = ["янв", "фев", "мар", "апр", "май", "июн", "июл", "авг", "сен", "окт", "ноя", "дек"]


def base(fig: go.Figure, left: int = 52, right: int = 16, top: int = LEGEND_SPACE, bottom: int = 44) -> go.Figure:
    fig.update_layout(
        template="none",
        font=dict(family=FONT, size=12, color=INK_2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=left, r=right, t=top, b=bottom),
        separators=", ",
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(family=FONT, size=12, color=INK)),
        legend=dict(orientation="h", xref="container", yref="container", x=0, xanchor="left", y=1,
                    yanchor="top", font=dict(size=12), itemclick=False, itemdoubleclick=False,
                    traceorder="normal"),
        showlegend=False,
        dragmode=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="", fixedrange=True,
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, showline=False, ticks="",
                     fixedrange=True, tickfont=dict(color=MUTED))
    return fig


def empty(message: str = "Нет карточек с вердиктом для выбранных фильтров") -> go.Figure:
    fig = base(go.Figure())
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=13, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


def percent_axis(fig: go.Figure, axis: str, top: float, title: str = "") -> None:
    ticks = [v for v in np.arange(0, 1.01, 0.25 if top > 0.5 else 0.1) if v <= top + 1e-9]
    update = fig.update_yaxes if axis == "y" else fig.update_xaxes
    update(range=[0, top], tickvals=ticks, ticktext=[ru_pct(v, 0) for v in ticks],
           title=dict(text=title, font=dict(color=MUTED)))


# ---------------------------------------------------------------- 1. Пропуски по неделям
def weekly(table: pd.DataFrame, overall: float) -> go.Figure:
    if table.empty:
        return empty()
    fig = base(go.Figure(), top=24)
    for reliable, color, name in ((True, MISSED, "неделя"), (False, AXIS, "мало инцидентов")):
        part = table.loc[table["reliable"] == reliable]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part["week"], y=part["share"], mode="markers", name=name,
            marker=dict(size=9, color=color, line=dict(color=SURFACE, width=2)),
            error_y=dict(type="data", symmetric=False, array=part["high"] - part["share"],
                         arrayminus=part["share"] - part["low"], color=color, thickness=1.5, width=0),
            customdata=np.column_stack([part["missed"], part["incidents"], part["low"], part["high"]]),
            hovertemplate="Неделя с %{x|%d.%m}: пропущено %{customdata[0]} из %{customdata[1]} "
                          "(%{y:.0%})<br>95 % интервал: %{customdata[2]:.0%}–%{customdata[3]:.0%}<extra></extra>",
        ))
    if np.isfinite(overall):
        fig.add_hline(y=overall, line=dict(color=NEUTRAL, width=1))
        fig.add_annotation(x=0, xref="paper", y=overall, text=f"в среднем {ru_pct(overall)}", showarrow=False,
                           xanchor="left", yanchor="bottom", font=dict(size=11, color=NEUTRAL))
    top = float(min(1.0, max(table["high"].max(), overall) + 0.05))
    percent_axis(fig, "y", top, "доля пропущенных инцидентов")
    fig.update_xaxes(type="date", tickformat="%d.%m", dtick=14 * 86400000, tick0="2025-09-01",
                     range=[table["week"].min() - pd.Timedelta(4, unit="D"),
                            table["week"].max() + pd.Timedelta(4, unit="D")])
    return fig


# ---------------------------------------------------------------- 2. Кривая поиска
def gains(curve: dict) -> go.Figure:
    if not curve:
        return empty()
    fig = base(go.Figure(), top=24, right=24)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color=AXIS, width=1.5),
                             hoverinfo="skip"))
    # Подпись диагонали — внизу слева под линией: там свободно при любой ширине экрана.
    fig.add_annotation(x=0.12, y=0.03, text="очередь без модели", showarrow=False, textangle=0,
                       xanchor="left", yanchor="bottom", font=dict(size=11, color=MUTED))
    fig.add_trace(go.Scatter(
        x=curve["queue"], y=curve["found"], mode="lines", line=dict(color=INK, width=2.5),
        hovertemplate="Разобрано %{x:.0%} очереди — найдено %{y:.0%} инцидентов<extra></extra>"))
    qx, qy = curve["threshold"]
    fig.add_trace(go.Scatter(x=[qx], y=[qy], mode="markers", hoverinfo="skip",
                             marker=dict(size=11, color=MISSED, line=dict(color=SURFACE, width=2))))
    fig.add_annotation(x=qx, y=qy, ax=40, ay=70, axref="pixel", ayref="pixel", showarrow=True,
                       arrowhead=0, arrowcolor=NEUTRAL, arrowwidth=1, xanchor="left", align="left",
                       bgcolor="rgba(255,255,255,0.9)",
                       text=(f"порог {str(THRESHOLD).replace('.', ',')}: модель поднимает {ru_pct(qx, 0)} очереди"
                             f"<br>и находит {ru_pct(qy, 0)} инцидентов, {ru_pct(1 - qy, 0)} — пропуск"),
                       font=dict(size=12, color=INK))
    percent_axis(fig, "x", 1.0, "доля очереди, разобранной сверху вниз по оценке модели")
    percent_axis(fig, "y", 1.02, "доля найденных реальных инцидентов")
    fig.update_xaxes(showgrid=True, gridcolor=GRID)
    return fig


# ---------------------------------------------------------------- 3. Слепые зоны
def categories(table: pd.DataFrame, overall: float) -> go.Figure:
    if table.empty:
        return empty("Мало реальных инцидентов для сравнения категорий")
    fig = base(go.Figure(), left=10, right=24, top=24)
    labels = list(table.index)
    fig.add_trace(go.Scatter(
        x=table["share"], y=labels, mode="markers",
        marker=dict(size=10, color=MISSED, line=dict(color=SURFACE, width=2)),
        error_x=dict(type="data", symmetric=False, array=table["high"] - table["share"],
                     arrayminus=table["share"] - table["low"], color=MISSED, thickness=1.5, width=0),
        customdata=np.column_stack([table["low"], table["high"]]),
        hovertemplate="%{y}: пропущено %{x:.0%}<br>95 % интервал: %{customdata[0]:.0%}–%{customdata[1]:.0%}"
                      "<extra></extra>",
    ))
    # Подпись «N из M» — в отдельной колонке справа, чтобы не лежать на чертах интервалов.
    for label, missed, incidents in zip(labels, table["missed"], table["incidents"]):
        fig.add_annotation(x=1.06, y=label, text=f"{ru_int(missed)} из {ru_int(incidents)}", showarrow=False,
                           xanchor="left", font=dict(size=12, color=INK_2))
    if np.isfinite(overall):
        fig.add_vline(x=overall, line=dict(color=NEUTRAL, width=1))
        fig.add_annotation(x=overall, y=1, yref="paper", text=f"в среднем {ru_pct(overall, 0)}",
                           showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=11, color=NEUTRAL))
    percent_axis(fig, "x", 1.3, "")
    fig.update_xaxes(showgrid=True, gridcolor=GRID, tickvals=[0, 0.25, 0.5, 0.75, 1],
                     ticktext=[ru_pct(v, 0) for v in (0, 0.25, 0.5, 0.75, 1)])
    fig.update_yaxes(showgrid=False, autorange="reversed", automargin=True, tickfont=dict(color=INK_2))
    return fig


# ---------------------------------------------------------------- 4. Уровень риска
def risk(rows: pd.DataFrame) -> go.Figure:
    if rows["cards"].sum() == 0:
        return empty()
    fig = base(go.Figure(), left=10, right=24, top=12, bottom=12)
    shares = rows["share"].fillna(0)
    fig.add_trace(go.Bar(
        x=shares, y=rows["row"], orientation="h", marker=dict(color=REAL), width=0.55,
        customdata=np.column_stack([rows["real"], rows["cards"], rows["of_all_real"]]),
        hovertemplate="%{y}<br>реальные инциденты: %{customdata[0]:,} из %{customdata[1]:,} карточек (%{x:.0%})"
                      "<br>это %{customdata[2]:.0%} всех реальных инцидентов<extra></extra>",
    ))
    for name, share, real, cards in zip(rows["row"], shares, rows["real"], rows["cards"]):
        fig.add_annotation(x=share, y=name, xanchor="left", showarrow=False, font=dict(size=12, color=INK_2),
                           text=f"  {ru_pct(share, 0)} · {ru_int(real)} из {ru_int(cards)}")
    fig.update_layout(barcornerradius=3)
    # Запас справа под подпись, чтобы она не обрезалась на узком экране.
    fig.update_xaxes(visible=False, range=[0, max(1.0, float(shares.max())) * 1.9])
    fig.update_yaxes(showgrid=False, autorange="reversed", automargin=True, tickfont=dict(color=INK_2))
    return fig


# ---------------------------------------------------------------- 5. Поток до и во время работы
def history(monthly: pd.DataFrame, train: tuple[str, str], work: tuple[str, str]) -> go.Figure:
    if monthly.empty:
        return empty("Положите iavd_ml_dataset.parquet в папку data/, чтобы сравнить поток с периодом обучения")
    fig = base(go.Figure(), right=110, top=24)
    x = pd.PeriodIndex(monthly["month"], freq="M").to_timestamp()
    for (start, end), label, fill in ((train, "обучение<br>(предположительно)", "rgba(137,135,129,0.12)"),
                                      (work, "работа<br>модели", "rgba(235,104,52,0.12)")):
        left = pd.Period(start, freq="M").to_timestamp() - pd.Timedelta(15, unit="D")
        right = pd.Period(end, freq="M").to_timestamp() + pd.Timedelta(15, unit="D")
        fig.add_vrect(x0=left, x1=right, fillcolor=fill, line_width=0, layer="below")
        fig.add_annotation(x=left + (right - left) / 2, y=1, yref="paper", text=label, showarrow=False,
                           yanchor="top", font=dict(size=11, color=MUTED))
    series = [("real_share", "доля реальных инцидентов", REAL), ("vendor_share", "заполнен «Вендор»", NEUTRAL)]
    for column, name, color in series:
        fig.add_trace(go.Scatter(
            x=x, y=monthly[column], mode="lines+markers", name=name, line=dict(color=color, width=2),
            marker=dict(size=6, color=color),
            customdata=[month_label(m) for m in monthly["month"]],
            hovertemplate="%{customdata}: " + name + " — %{y:.0%}<extra></extra>"))
        last = monthly[column].dropna()
        if len(last):
            fig.add_annotation(x=x[last.index[-1]], y=last.iloc[-1], text=f"  {name}", showarrow=False,
                               xanchor="left", font=dict(size=12, color=INK_2))
    months = pd.date_range(x.min(), x.max(), freq="3MS")
    fig.update_xaxes(type="date", tickvals=list(months),
                     ticktext=[SHORT_MONTHS[d.month - 1] + (f"<br>{d.year}" if d.month <= 3 or i == 0 else "")
                               for i, d in enumerate(months)])
    percent_axis(fig, "y", 1.0, "")
    return fig
