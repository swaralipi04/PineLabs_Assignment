"""Visualisation agent (deterministic): picks a chart from the result shape."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from core.analysis import DriverReport

# Pine Labs green palette: BLUE=main series, RED=negative/focus (deep pine), GREY=baseline (pale green)
BLUE, RED, GREY, GREEN = "#2e8b5e", "#12332b", "#a9d6bb", "#5fbf8a"


def _is_rate(col: str) -> bool:
    return any(k in col.lower() for k in ("rate", "share", "pct"))


def _layout(fig, title):
    fig.update_layout(title=title, margin=dict(l=10, r=10, t=50, b=10), height=380, legend_title_text="",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(46,139,94,0.15)")
    return fig


def auto_chart(df: pd.DataFrame, title: str = "", shade: tuple | None = None):
    if df is None or df.empty or len(df.columns) < 2:
        return None
    df = df.copy()
    date_cols = [c for c in df.columns if pd.api.types.is_datetime64_any_dtype(df[c]) or "date" in c.lower() or c.lower() in ("week_start", "day", "week")]
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and c not in date_cols]
    cat_cols = [c for c in df.columns if c not in num_cols and c not in date_cols]
    if not num_cols:
        return None
    y = next((c for c in num_cols if _is_rate(c)), num_cols[0])
    if date_cols:
        x = date_cols[0]
        df[x] = pd.to_datetime(df[x])
        color = cat_cols[0] if cat_cols and df[cat_cols[0]].nunique() <= 8 else None
        fig = px.line(df.sort_values(x), x=x, y=y, color=color, markers=True, color_discrete_sequence=[BLUE, RED, GREEN, "#1f6b4a", "#8fd3ad"])
        if shade:
            fig.add_vrect(x0=pd.Timestamp(shade[0]), x1=pd.Timestamp(shade[1]) + pd.Timedelta(hours=23), fillcolor=GREEN, opacity=0.15, line_width=0,
                          annotation_text="focus period", annotation_position="top left")
    elif cat_cols:
        x = cat_cols[0]
        d = df.sort_values(y, ascending=True).tail(20)
        fig = px.bar(d, x=y, y=x, orientation="h", color_discrete_sequence=[BLUE])
    else:
        return None
    if _is_rate(y):
        fig.update_yaxes(tickformat=".1%") if date_cols else fig.update_xaxes(tickformat=".1%")
    return _layout(fig, title or y.replace("_", " ").title())


def driver_chart(rep: DriverReport, dim=None, title: str | None = None):
    """Contribution of each segment of a dimension, split into rate vs mix effect."""
    dim = dim or rep.primary
    if not dim:
        return None
    t = dim.table.copy()
    t = t.reindex(t["contribution"].abs().sort_values(ascending=False).index).head(10).iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=t["segment"], x=t["rate_effect"] * 100, name="Rate effect (segment got better/worse)", orientation="h", marker_color=RED)
    fig.add_bar(y=t["segment"], x=t["mix_effect"] * 100, name="Mix effect (traffic share moved)", orientation="h", marker_color=GREY)
    fig.update_layout(barmode="relative")
    fig.update_xaxes(title="Contribution to change (percentage points)", zeroline=True, zerolinecolor="#12332b")
    return _layout(fig, title or f"What drove the {rep.delta * 100:+.2f} pt change - by {dim.label.lower()}")


def segment_compare_chart(rep: DriverReport):
    """Base vs focus success rate and traffic share per segment (reveals mix shifts)."""
    if not rep.primary:
        return None
    t = rep.primary.table.sort_values("focus_n", ascending=False).head(10)
    fig = go.Figure()
    fig.add_bar(x=t["segment"], y=t["base_rate"], name=f"Success rate - {rep.base.start}..{rep.base.end}", marker_color=GREY)
    fig.add_bar(x=t["segment"], y=t["focus_rate"], name=f"Success rate - {rep.focus.start}..{rep.focus.end}", marker_color=BLUE)
    fig.add_scatter(x=t["segment"], y=t["base_share"], name="Traffic share (before)", mode="markers", marker=dict(symbol="diamond-open", size=11, color="#12332b"), yaxis="y2")
    fig.add_scatter(x=t["segment"], y=t["focus_share"], name="Traffic share (after)", mode="markers", marker=dict(symbol="diamond", size=11, color=RED), yaxis="y2")
    lo = max(0, min(t["base_rate"].min(), t["focus_rate"].min()) - 0.05)
    fig.update_layout(barmode="group", yaxis=dict(tickformat=".0%", range=[lo, 1], title="Success rate"),
                      yaxis2=dict(overlaying="y", side="right", tickformat=".0%", title="Traffic share", showgrid=False))
    return _layout(fig, f"{rep.primary.label}: success rate and traffic share, before vs after")
