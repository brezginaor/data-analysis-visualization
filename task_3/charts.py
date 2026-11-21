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
def feature_label(name: str) -> str:
    """Короткие переносы только на оси; полное имя остаётся в hover."""
    if name == "Оценка модели (лог. шансов)":
        return "Итог (logit)"
    if name.startswith("остальные "):
        return name.replace(" признаков", "<br>признаков")
    if len(name) <= 17:
        return name
    if " " in name:
        first, rest = name.split(" ", 1)
        return f"{first}<br>{rest}"
    if "_" in name:
        first, rest = name.split("_", 1)
        return f"{first}_<br>{rest}"
    return name


def waterfall(table: pd.DataFrame, proba: float) -> go.Figure:
    if table.empty:
        return empty()
    labels = [feature_label(str(n)) for n in table["feature"]]
    base_value = float(table.iloc[0]["value"])
    final_value = float(table.iloc[-1]["value"])

    # Все промежуточные значения учитываются в диапазоне оси. Нуль не нужен:
    # показываем изменения logit, а не длинные полосы от нуля до базы/итога.
    relative = table.loc[table["measure"] == "relative", "value"].astype(float)
    levels = np.r_[base_value, base_value + relative.cumsum().to_numpy()]
    span = max(float(levels.max() - levels.min()), 0.5)
    padding = max(span * 0.12, 0.25)

    fig = base(go.Figure(), left=180, right=54, top=49, bottom=50)
    fig.add_trace(go.Waterfall(
        orientation="h", y=labels, x=table["value"], measure=table["measure"],
        increasing=dict(marker=dict(color=TO_INCIDENT)),
        decreasing=dict(marker=dict(color=TO_LEGIT)),
        # Plotly использует абсолютную базу и итог для расчёта ступеней.
        # Их собственные полосы скрываем: вместо них ниже ставим метки.
        totals=dict(marker=dict(color="rgba(0,0,0,0)",
                                line=dict(color="rgba(0,0,0,0)", width=0))),
        connector=dict(line=dict(color=AXIS, width=1, dash="dot")),
        customdata=table["feature"],
        hovertemplate="%{customdata}: %{x:+.3f} лог. шансов<extra></extra>",
        textposition="outside",
        text=[f"{v:+.2f}" if m == "relative" else ""
              for v, m in zip(table["value"], table["measure"])],
        textfont=dict(size=10, color=INK_2),
    ))
    fig.add_trace(go.Scatter(
        x=[base_value, final_value], y=[labels[0], labels[-1]],
        mode="markers+text", showlegend=False,
        marker=dict(symbol="line-ns", size=23, color=INK_2,
                    line=dict(width=3, color=INK_2)),
        text=[ru_num(base_value), ru_num(final_value)],
        textposition="middle right", textfont=dict(size=11, color=INK),
        customdata=["Базовая линия", "Итог модели"],
        hovertemplate="%{customdata}: %{x:.3f} лог. шансов<extra></extra>",
    ))
    outcome = "в приоритете" if proba >= THRESHOLD else "отложена"
    fig.add_annotation(x=1, xref="paper", y=1.045, yref="paper", xanchor="right", yanchor="bottom",
                       text=f"Вероятность: {ru_proba(proba)} · {outcome}", showarrow=False,
                       font=dict(size=12, color=INK))
    fig.update_xaxes(title=dict(text="Логарифм шансов (logit)", font=dict(color=MUTED)),
                     range=[float(levels.min()) - padding, float(levels.max()) + padding * 1.6],
                     showgrid=True, gridcolor=GRID, automargin=True)
    fig.update_yaxes(showgrid=False, autorange="reversed", automargin=True,
                     tickfont=dict(color=INK_2, size=11))
    fig.update_layout(bargap=0.20)
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
    fig = base(go.Figure(), left=56, right=20, top=56, bottom=46)
    for outcome in ("TP", "FP", "FN", "TN"):
        part = table.loc[table["outcome"] == outcome]
        if part.empty:
            continue
        fig.add_trace(go.Bar(
            x=[feature_label(str(name)) for name in part["feature"]],
            y=part["mean_shap"], name=outcome,
            marker=dict(color=OUTCOME_COLORS[outcome]), width=0.7,
            customdata=[[str(name), OUTCOME_NAMES[outcome]] for name in part["feature"]],
            hovertemplate="%{customdata[0]}<br>%{customdata[1]}: %{y:+.2f} лог. шанса<extra></extra>",
        ))
    fig.add_hline(y=0, line=dict(color=INK, width=1))
    fig.update_layout(barmode="group", bargap=0.35, bargroupgap=0.08,
                      legend=dict(orientation="h", x=0, y=1, yanchor="bottom",
                                  font=dict(size=12), itemclick=False, itemdoubleclick=False),
                      showlegend=True)
    fig.update_yaxes(title=dict(text="Средний SHAP, лог. шансов", font=dict(color=MUTED)),
                     automargin=True)
    fig.update_xaxes(automargin=True, tickfont=dict(color=INK_2, size=10), tickangle=0)
    return fig


# ---------------------------------------------------------------- 4. Один сомнительный случай: две оценки и порог

def doubt_comparison(proba: float, conditional: float, entity_id: int) -> go.Figure:
    """Показывает арифметическое исключение SHAP-вкладов, НЕ ablation признаков."""
    fig = base(go.Figure(), left=20, right=48, top=42, bottom=34)
    fig.add_trace(go.Bar(
        x=[proba, conditional], y=["Оценка модели", "Пересчёт SHAP без 3 вкладов"],
        orientation="h", width=0.52,
        marker=dict(color=[DOUBT, TO_LEGIT]),
        text=[ru_pct(proba, 2), ru_pct(conditional, 2)],
        textposition="outside", textfont=dict(size=13, color=INK),
        cliponaxis=False,
        customdata=["Исходный прогноз модели",
                    "Только арифметический пересчёт вкладов; модель повторно не запускалась"],
        hovertemplate=f"Карточка {entity_id} (FP)<br>%{{y}}: %{{x:.2%}}<br>%{{customdata}}<extra></extra>",
    ))
    fig.add_vline(x=THRESHOLD, line=dict(color=INK_2, width=1.5, dash="dash"))
    fig.add_annotation(
        x=THRESHOLD, xref="x", y=1.05, yref="paper",
        text=f"Порог {ru_pct(THRESHOLD, 2)}", showarrow=False,
        xanchor="center", yanchor="bottom", font=dict(color=INK_2, size=11),
    )
    fig.update_layout(bargap=0.5)
    fig.update_yaxes(showgrid=False, autorange="reversed", automargin=True,
                     tickfont=dict(color=INK_2, size=12))
    fig.update_xaxes(range=[0, max(proba, conditional, THRESHOLD) * 1.28],
                     tickformat=".1%", title=None, automargin=True)
    return fig
