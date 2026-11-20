"""Plotly-фигуры дашборда задания 3. Каждая функция: готовые данные -> фигура."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from loader import THRESHOLD, ru_num, ru_pct, ru_proba

FONT = "system-ui, -apple-system, 'Segoe UI', Roboto, Arial, sans-serif"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#ecebe6"
AXIS = "#c3c2b7"
SURFACE = "#ffffff"

# Один цвет — один смысл, как в заданиях 1 и 2:
# оранжевый — толчок «к инциденту», серый — «к легитимному», тёмно-красный — сомнение.
TO_INCIDENT = "#eb6834"
TO_LEGIT = "#6f6d68"
DOUBT = "#b42318"
PLOT_CONFIG = {"displayModeBar": False, "responsive": True, "scrollZoom": False}


def base(fig: go.Figure, left: int = 52, right: int = 16, top: int = 12, bottom: int = 44) -> go.Figure:
    fig.update_layout(
        template="none",
        font=dict(family=FONT, size=12, color=INK_2),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=left, r=right, t=top, b=bottom),
        separators=", ",
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=AXIS, font=dict(family=FONT, size=12, color=INK)),
        showlegend=False,
        dragmode=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showline=False, ticks="", fixedrange=True,
                     tickfont=dict(color=MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRID, zeroline=False, showline=False, ticks="",
                     fixedrange=True, tickfont=dict(color=MUTED))
    return fig


def empty(message: str = "Нет карточек для выбранных фильтров") -> go.Figure:
    fig = base(go.Figure())
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font=dict(size=13, color=MUTED))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig


# ---------------------------------------------------------------- 1. Водопад карточки
def waterfall(table: pd.DataFrame, proba: float) -> go.Figure:
    if table.empty:
        return empty()
    fig = base(go.Figure(), left=10, right=24, top=30)
    fig.add_trace(go.Waterfall(
        x=table["feature"], measure=table["measure"], y=table["value"],
        increasing=dict(marker=dict(color=TO_INCIDENT)),
        decreasing=dict(marker=dict(color=TO_LEGIT)),
        totals=dict(marker=dict(color=INK)),
        connector=dict(line=dict(color=AXIS, width=1, dash="dot")),
        hovertemplate="%{x}: %{y:+.3f} лог. шансов<extra></extra>",
        textposition="outside",
        text=[f"{v:+.2f}" if m == "relative" else f"{v:.2f}" for v, m in zip(table["value"], table["measure"])],
        textfont=dict(size=11, color=INK_2),
    ))
    outcome = "инцидент" if proba >= THRESHOLD else "отложена"
    fig.add_annotation(x=1, xref="paper", y=1, yref="paper", xanchor="right", yanchor="bottom",
                       text=f"sigmoid(итог) = {ru_proba(proba)} → {outcome}", showarrow=False,
                       font=dict(size=12, color=INK))
    fig.update_yaxes(title=dict(text="логарифм шансов «инцидента»", font=dict(color=MUTED)), automargin=True)
    fig.update_xaxes(automargin=True, tickfont=dict(color=INK_2, size=11))
    fig.update_layout(bargap=0.25)
    return fig


# ---------------------------------------------------------------- 2. Опора потока
def mean_abs(table: pd.DataFrame) -> go.Figure:
    if table.empty:
        return empty()
    fig = base(go.Figure(), left=10, right=90, top=12, bottom=12)
    fig.add_trace(go.Bar(
        x=table["mean_abs"], y=table["feature"], orientation="h", width=0.55,
        marker=dict(color=TO_INCIDENT), opacity=0.9,
        customdata=table["share"],
        hovertemplate="%{y}: средний вклад %{x:.2f} лог. шанса<br>%{customdata:.0%} суммы топ-10<extra></extra>",
    ))
    for name, value, share in zip(table["feature"], table["mean_abs"], table["share"]):
        fig.add_annotation(x=value, y=name, xanchor="left", showarrow=False,
                           font=dict(size=12, color=INK_2), text=f"  {ru_num(value)}")
    fig.update_layout(barcornerradius=3)
    fig.update_xaxes(title=dict(text="средний |SHAP|, лог. шансов", font=dict(color=MUTED)),
                     range=[0, float(table["mean_abs"].max()) * 1.25], automargin=True)
    fig.update_yaxes(showgrid=False, autorange="reversed", automargin=True, tickfont=dict(color=INK_2))
    return fig


# ---------------------------------------------------------------- 3. Вклады по исходам
OUTCOME_NAMES = {"TP": "инцидент,<br>поднята", "FP": "не инцидент,<br>поднята",
                 "FN": "инцидент,<br>отложена", "TN": "не инцидент,<br>отложена"}
OUTCOME_COLORS = {"TP": TO_INCIDENT, "FP": DOUBT, "FN": "#7c3aed", "TN": TO_LEGIT}


def signed_outcomes(table: pd.DataFrame) -> go.Figure:
    if table.empty:
        return empty()
    fig = base(go.Figure(), left=10, right=24, top=70)
    for outcome in ("TP", "FP", "FN", "TN"):
        part = table.loc[table["outcome"] == outcome]
        if part.empty:
            continue
        fig.add_trace(go.Bar(
            x=part["feature"], y=part["mean_shap"], name=outcome,
            marker=dict(color=OUTCOME_COLORS[outcome]), width=0.7,
            customdata=[OUTCOME_NAMES[o] for o in part["outcome"]],
            hovertemplate="%{x}<br>%{customdata}: %{y:+.2f} лог. шанса<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=INK, width=1))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08,
                      legend=dict(orientation="h", x=0, y=1, yanchor="bottom",
                                  font=dict(size=12), itemclick=False, itemdoubleclick=False),
                      showlegend=True)
    fig.update_yaxes(title=dict(text="средний SHAP: «+» к инциденту, «−» к легитимному",
                                font=dict(color=MUTED)), automargin=True)
    fig.update_xaxes(automargin=True, tickfont=dict(color=INK_2, size=11))
    return fig


# ---------------------------------------------------------------- 4. Роль «Алерта» в потоке
def alert_scatter(flow: pd.DataFrame, alert_column: str = "Алерт") -> go.Figure:
    if flow.empty or alert_column not in flow.columns:
        return empty()
    fig = base(go.Figure(), left=52, right=24, top=70, bottom=48)
    names = {"TP": "инцидент, поднята", "FP": "не инцидент, поднята",
             "FN": "инцидент, отложена", "TN": "не инцидент, отложена"}
    colors = {"TP": TO_INCIDENT, "FP": DOUBT, "FN": "#7c3aed", "TN": TO_LEGIT}
    for outcome in ("TP", "FP", "FN", "TN"):
        part = flow.loc[flow["outcome"] == outcome]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(
            x=part[alert_column].clip(lower=-0.02), y=part["proba"].clip(lower=5e-7),
            mode="markers", name=names[outcome],
            marker=dict(size=9, color=colors[outcome], line=dict(color=SURFACE, width=1.5), opacity=0.85),
            customdata=part["EntityId"],
            hovertemplate="Карточка %{customdata}<br>вклад «" + alert_column + "»: %{x:+.2f}"
                          "<br>оценка: %{y:.4f}<extra></extra>",
        ))
    logit_threshold = np.log(THRESHOLD / (1 - THRESHOLD))
    fig.add_hline(y=THRESHOLD, line=dict(color=INK, width=1, dash="dash"))
    fig.add_annotation(x=0, xref="paper", y=THRESHOLD, text=f"порог {ru_num(THRESHOLD, 4)}",
                       showarrow=False, xanchor="left", yanchor="bottom", font=dict(size=11, color=MUTED))
    fig.update_layout(legend=dict(orientation="h", x=0, y=1, yanchor="bottom", font=dict(size=12),
                                  itemclick=False, itemdoubleclick=False), showlegend=True)
    fig.update_yaxes(type="log", title=dict(text="оценка модели (лог. шкала)", font=dict(color=MUTED)))
    fig.update_xaxes(title=dict(text=f"вклад признака «{alert_column}», лог. шансов", font=dict(color=MUTED)))
    return fig
