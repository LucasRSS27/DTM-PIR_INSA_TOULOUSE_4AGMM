from shiny import App, ui, reactive, render
from shinywidgets import output_widget, render_widget

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import torch
import pandas as pd
import sys
import io
from wordcloud import WordCloud
from PIL import Image

from dtm_core import (
    DynamicTopicModel,
    load_csv_data,
    get_smoothed_beta,
    get_smoothed_alpha,
)
import dtm_core

# ══════════════════════════════════════════════════════════════════
# GLOBAL CSS (MODERN SaaS & BADGES)
# ══════════════════════════════════════════════════════════════════
GLOBAL_CSS = '''
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&display=swap');

:root {
    --bg: #f8fafc;
    --sidebar-bg: #ffffff;
    --card: #ffffff;
    --text: #0f172a;
    --muted: #64748b;
    --accent: #0ea5e9;
    --soft: #f1f5f9;
    --border: #e2e8f0;
    --mono: 'IBM Plex Mono', monospace;
    --radius-input: 10px;
    --radius-card: 16px;
    --radius-btn: 12px;
}

body {
    background: var(--bg);
    font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'SF Pro Display', sans-serif;
    color: var(--text);
}

.sidebar {
    background: var(--sidebar-bg) !important;
    border-right: 1px solid var(--border) !important;
    padding: 16px 12px !important; /* Reduced padding for a more compact look */
}

/* Tighten spacing between sidebar elements */
.sidebar hr {
    margin: 8px 0 !important; 
}

.sidebar .section-title {
    font-size: 0.68rem;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 8px; /* Reduced from 16px */
    margin-top: 4px;
}

.sidebar .shiny-input-container {
    margin-bottom: 10px !important; /* Tighter input grouping */
}
/* ── CARDS ────────────────────────────────────────────────────── */
.dtm-card {
    background: var(--card);
    border-radius: var(--radius-card);
    padding: 28px;
    border: 1px solid var(--border);
    box-shadow: 0 1px 2px rgba(15,23,42,0.03), 0 2px 8px rgba(15,23,42,0.02);
    margin-bottom: 20px;
    transition: box-shadow 0.2s ease;
}
.dtm-card:hover {
    box-shadow: 0 2px 6px rgba(15,23,42,0.06), 0 4px 16px rgba(15,23,42,0.04);
}

.section-title {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.07em;
    margin-bottom: 16px;
}

/* ── INPUTS ───────────────────────────────────────────────────── */
input, select {
    border-radius: var(--radius-input) !important;
    border: 1px solid var(--border) !important;
    padding: 8px 12px;
    font-size: 13px;
    transition: all 0.18s ease;
    background: #ffffff !important;
}
input:focus, select:focus {
    border-color: var(--accent) !important;
    box-shadow: 0 0 0 3px rgba(14,165,233,0.12) !important;
    outline: none !important;
}
input[type="range"] {
    accent-color: var(--accent);
    cursor: pointer;
}

/* ── FILE INPUT ───────────────────────────────────────────────── */
/* Shiny renders file input as .input-group > .btn-file + .form-control */
input[type=file] { display: none !important; }

.btn-file {
    border-radius: var(--radius-input) !important;
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    padding: 7px 16px !important;
    font-size: 12px !important;
    font-weight: 600 !important;
    cursor: pointer !important;
    transition: background 0.18s ease, border-color 0.18s ease, transform 0.15s ease !important;
    box-shadow: 0 1px 2px rgba(15,23,42,0.05) !important;
    white-space: nowrap !important;
}
.btn-file:hover {
    background: var(--soft) !important;
    border-color: #cbd5e1 !important;
    transform: translateY(-1px) !important;
}
.shiny-input-container .input-group {
    display: flex;
    align-items: center;
    gap: 10px;
    flex-wrap: nowrap;
}
.shiny-input-container .input-group .form-control[readonly] {
    font-size: 11px !important;
    color: var(--muted) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-input) !important;
    background: var(--soft) !important;
    padding: 7px 10px !important;
    height: auto !important;
    flex: 1;
    min-width: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    box-shadow: none !important;
}
/* ── FILE UPLOAD PROGRESS BAR ────────────────────────────────── */
#shiny-progress-output .progress,
.shiny-file-input-progress .progress,
.progress {
    height: 30px !important;  /* ← ajuste cette valeur selon ton goût */
    border-radius: 999px !important;
    background: var(--soft) !important;
    overflow: hidden !important;
}
.progress .progress-bar {
    background: var(--accent) !important;
    border-radius: 999px !important;
    transition: width 0.25s ease !important;
}
/* ── NAVBAR TABS ─────────────────────────────────────────────── */
.navbar {
    border: 1px solid var(--border) !important;
    background: #ffffff !important;
    border-radius: var(--radius-card) !important; /* Rounded corners */
    margin: 12px 16px 0px 16px !important; /* Margins to detach it from the edges and reveal corners */
    box-shadow: 0 1px 2px rgba(15,23,42,0.03), 0 2px 8px rgba(15,23,42,0.02) !important; /* Match card shadow */
}
.nav-tabs {
    border-bottom: 1px solid var(--border) !important;
    background: #ffffff !important;
}
.nav-tabs { border-bottom: none !important; padding: 6px 8px 0 !important; }
.nav-tabs .nav-link {
    border: none !important;
    border-radius: 10px !important;
    color: var(--muted) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    padding: 7px 18px !important;
    margin: 0 2px !important;
    transition: all 0.15s ease !important;
    background: transparent !important;
}
.nav-tabs .nav-link:hover {
    color: var(--text) !important;
    background: var(--soft) !important;
}
.nav-tabs .nav-link.active {
    background: var(--accent) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(14,165,233,0.25) !important;
}

/* ── SHINY PROGRESS MODAL ────────────────────────────────────── */
.shiny-progress-container {
    bottom: 24px !important;
    right: 24px !important;
    top: auto !important;
    width: 300px !important;
}
.shiny-progress {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: 14px !important;
    box-shadow: 0 8px 32px rgba(15,23,42,0.12) !important;
    padding: 16px 20px !important;
    margin-top: 8px !important;
}
.shiny-progress .shiny-progress-message {
    font-size: 11px !important;
    color: var(--muted) !important;
    font-family: var(--mono) !important;
    margin-bottom: 8px !important;
    font-weight: 600 !important;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
.shiny-progress .progress {
    height: 5px !important;
    border-radius: 999px !important;
    background: var(--soft) !important;
    overflow: hidden !important;
    margin: 0 !important;
}
.shiny-progress .progress-bar {
    background: var(--accent) !important;
    border-radius: 999px !important;
    transition: width 0.25s ease !important;
}

/* ── TOAST NOTIFICATIONS ─────────────────────────────────────── */
.shiny-notification {
    border-radius: 12px !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 8px 24px rgba(15,23,42,0.1) !important;
    padding: 14px 18px !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    background: #ffffff !important;
    color: var(--text) !important;
}
.shiny-notification-message { border-left: 3px solid #16a34a !important; }
.shiny-notification-warning { border-left: 3px solid #f59e0b !important; }
.shiny-notification-error   { border-left: 3px solid #ef4444 !important; }
.shiny-notification-close { color: var(--muted) !important; font-size: 16px !important; }

/* ── BADGES ───────────────────────────────────────────────────── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 12px;
    border-radius: 999px;
    font-size: 0.68rem;
    font-weight: 700;
    margin-bottom: 16px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
}
.status-badge.trained  { background: #dcfce7; color: #15803d; }
.status-badge.filtered { background: #fef9c3; color: #a16207; }
.status-badge.pending  { background: var(--soft); color: #475569; }

.date-badge {
    background: #ffffff;
    border: 1px solid var(--border);
    padding: 3px 9px;
    border-radius: 7px;
    font-family: var(--mono);
    font-size: 0.68rem;
    font-weight: 600;
    color: var(--text);
    box-shadow: 0 1px 2px rgba(15,23,42,0.04);
}

/* ── BUTTONS ──────────────────────────────────────────────────── */
.btn-primary {
    background: var(--accent) !important;
    border: none !important;
    border-radius: var(--radius-btn) !important;
    padding: 10px 14px;
    font-weight: 600;
    font-size: 13px;
    box-shadow: 0 1px 3px rgba(14,165,233,0.25);
    transition: all 0.15s ease !important;
}
.btn-primary:hover {
    background: #0284c7 !important;
    box-shadow: 0 3px 10px rgba(14,165,233,0.35) !important;
    transform: translateY(-1px);
    
}
.btn-filter {
    background: #ffffff !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-btn) !important;
    font-weight: 500;
    font-size: 13px;
    transition: all 0.15s ease !important;
}
.btn-filter:hover {
    background: var(--soft) !important;
    border-color: #cbd5e1 !important;
    transform: translateY(-1px);
    color: var(--text) !important;
}
button { transition: all 0.15s ease !important; }

/* ── STAT CARDS ───────────────────────────────────────────────── */
.stats-row {
    display: flex;
    gap: 14px;
    margin-bottom: 24px;
    flex-wrap: wrap;
}
.stat-card {
    padding: 20px 20px 18px 24px;
    border-radius: 14px;
    background: #ffffff;
    border: 1px solid var(--border);
    border-left: 3px solid transparent;
    flex: 1;
    min-width: 160px;
    transition: box-shadow 0.2s ease, transform 0.2s ease;
    position: relative;
    overflow: hidden;
}
.stat-card:hover {
    box-shadow: 0 4px 16px rgba(15,23,42,0.07);
    transform: translateY(-1px);
}
.stat-card.docs   { border-left-color: #0ea5e9; }
.stat-card.vocab  { border-left-color: #8b5cf6; }
.stat-card.avgl   { border-left-color: #16a34a; }

.stat-card-label {
    font-size: 0.68rem;
    color: var(--muted);
    text-transform: uppercase;
    font-weight: 700;
    letter-spacing: 0.06em;
    margin-bottom: 6px;
}
.stat-card-value {
    font-size: 1.75rem;
    font-weight: 800;
    color: var(--text);
    line-height: 1;
    letter-spacing: -0.03em;
    animation: countUp 0.4s ease;
}
.stat-card.docs  .stat-card-value { color: #0284c7; }
.stat-card.vocab .stat-card-value { color: #7c3aed; }
.stat-card.avgl  .stat-card-value { color: #15803d; }

.stat-card-sub {
    font-size: 0.72rem;
    color: #94a3b8;
    margin-top: 5px;
    font-weight: 500;
}

@keyframes countUp {
    from { opacity: 0; transform: translateY(6px); }
    to   { opacity: 1; transform: translateY(0);   }
}

/* ── FREX SLIDER ──────────────────────────────────────────────── */
.frex-block {
    min-width: 220px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
}
.frex-label-top {
    font-size: 0.72rem;
    font-weight: 700;
    color: var(--text);
    letter-spacing: 0.01em;
    margin-bottom: 2px;
}
.frex-label-desc {
    font-size: 0.67rem;
    color: var(--muted);
    margin-bottom: 6px;
}
.frex-minmax {
    display: flex;
    justify-content: space-between;
    font-size: 0.68rem;
    font-weight: 600;
    color: #94a3b8;
    margin-top: 2px;
    padding: 0 2px;
    letter-spacing: 0.02em;
}
.frex-minmax span.active-end { color: var(--accent); }

/* ── TOPIC CONTROL BAR ────────────────────────────────────────── */
.topic-ctrl-selects {
    display: flex;
    align-items: flex-end;
    gap: 16px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.topic-ctrl-selects .shiny-input-container {
    margin-bottom: 0 !important;
}

/* ── PROGRESS ─────────────────────────────────────────────────── */
.dtm-progress-track {
    height: 5px;
    background: var(--soft);
    border-radius: 999px;
    overflow: hidden;
    margin-top: 8px;
}
'''

# ══════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════
app_ui = ui.page_sidebar(

    ui.sidebar(
        ui.tags.style(GLOBAL_CSS),

        ui.tags.div(
            ui.h3("DTM Analytics", style="font-weight:800; color:var(--text); letter-spacing:-0.5px; margin:0;"),
            style="padding: 10px 0 25px 0;"
        ),
        ui.hr(),

        ui.h6("DATA", class_="section-title"),
        ui.input_file("file", "", accept=[".csv"]),
        ui.output_ui("column_selectors"),
        ui.hr(),

        ui.h6("FILTERS", class_="section-title"),
        ui.input_select("language", "Language", choices={"english": "English", "french": "Français"}, selected="english"),
        ui.input_select("granularity", "Time period", choices={"Y": "Year", "Q": "Quarter", "M": "Month", "W": "Week"}, selected="Y"),
        ui.output_ui("time_range_ui"),
        ui.hr(),

        ui.h6("TRAINING", class_="section-title"),
        ui.input_numeric("topics", "Topic count", 5, min=2, max=20),
        ui.input_numeric("epochs", "Training steps", 500, min=10),
        ui.hr(),

        ui.input_action_button("apply_filter", "⚙  Apply Filter", class_="btn-filter w-100 mb-2"),
        ui.input_action_button("run", "▶  Train Model", class_="btn-primary w-100"),
        ui.hr(),

        ui.output_ui("status"),
        ui.output_ui("progress_bar"),

        width=280,
    ),

    ui.page_navbar(
        ui.nav_panel("Overview",
            ui.output_ui("overview_stats"),
            ui.tags.div(
                ui.tags.p("Data overview", class_="section-title"),
                output_widget("corpus"),
                class_="dtm-card"
            ),
        ),

        ui.nav_panel("Topics",
            ui.tags.div(
                ui.tags.p("Word trends", class_="section-title"),
                ui.tags.div(
                    ui.tags.div(ui.input_select("topic_id", "Topic", choices=["0"]), style="width:120px;"),
                    ui.tags.div(ui.input_numeric("n_words_plot", "Top words", 8, min=2, max=30), style="width:100px;"),
                    ui.tags.div(
                        ui.tags.div("Word focus", class_="frex-label-top"),
                        ui.tags.div("From common to distinctive words", class_="frex-label-desc"),
                        ui.input_slider("specificity", None, min=0, max=1, value=0, step=0.1),
                        ui.tags.div(
                            ui.tags.span("Common"),
                            ui.tags.span("Distinctive"),
                            class_="frex-minmax"
                        ),
                        class_="frex-block"
                    ),
                    class_="topic-ctrl-selects"
                ),
                output_widget("topics_plot"),
                class_="dtm-card"
            ),
            ui.tags.div(
                ui.tags.p("Words at a glance", class_="section-title"),
                ui.tags.div(
                    ui.input_select("time_slice", "Period", choices=["—"]),
                    class_="topic-ctrl-selects"
                ),
                output_widget("top_words_time"),
                class_="dtm-card"
            ),
        ),

        ui.nav_panel("Evolution",
            ui.tags.div(
                ui.tags.p("Topic share over time", class_="section-title"),
                output_widget("evolution"),
                class_="dtm-card"
            ),
            ui.tags.div(
                ui.tags.p("Word clouds by topic", class_="section-title"),
                ui.output_ui("wordclouds"),
                class_="dtm-card"
            ),
        ),

        ui.nav_panel("Diagnostics",
            ui.layout_column_wrap(
                ui.tags.div(
                    ui.tags.p("Training loss", class_="section-title"),
                    output_widget("elbo"),
                    class_="dtm-card"
                ),
                ui.tags.div(
                    ui.tags.p("Run details", class_="section-title"),
                    ui.output_ui("convergence_status"),
                    ui.hr(),
                    ui.tags.p("Learning rate", style="font-size:0.7rem; color:var(--muted); margin-bottom:4px; text-transform:uppercase; font-weight:600;"),
                    ui.output_ui("lr_display"),
                    class_="dtm-card"
                ),
                width=1/2
            )
        ),

        id="main_tabs",
    )
)

# ══════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════
COLORS_10 = ["#0ea5e9", "#16a34a", "#f97316", "#8b5cf6", "#ef4444", "#06b6d4", "#84cc16", "#f59e0b", "#a855f7", "#ec4899"]
COLORS_20 = COLORS_10 + ["#7dd3fc", "#86efac", "#fdba74", "#c4b5fd", "#fca5a5", "#67e8f9", "#d9f99d", "#fde68a", "#e9d5ff", "#fbcfe8"]

def _empty_fig(msg="Train a model to see results"):
    fig = go.Figure()
    fig.add_annotation(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False, font=dict(size=13, color="#9ca3af", family="IBM Plex Mono"))
    fig.update_layout(template="plotly_white", height=340, xaxis=dict(visible=False), yaxis=dict(visible=False), margin=dict(l=30, r=30, t=30, b=30))
    return fig

def _smooth(arr, window):
    if window <= 1 or len(arr) < window: return arr
    return pd.Series(arr).rolling(window, center=True, min_periods=1).mean().values

def _adaptive_smooth_window(T):
    if T >= 60: return 7
    if T >= 30: return 5
    if T >= 15: return 3
    return 1

# ══════════════════════════════════════════════════════════════════
# SERVER
# ══════════════════════════════════════════════════════════════════
def server(input, output, session):

    filter_state  = reactive.Value(None)
    model_state   = reactive.Value(None)
    conv_msg      = reactive.Value("")
    loading       = reactive.Value(False)
    prog_pct      = reactive.Value(0)
    prog_msg      = reactive.Value("")
    filter_dirty  = reactive.Value(False)

    @output
    @render.ui
    def column_selectors():
        f = input.file()
        if f is None: return ui.div()
        try:
            df = pd.read_csv(f[0]["datapath"], nrows=5)
            cols = list(df.columns)
            return ui.div(
                ui.hr(),
                ui.input_select("col_content", "Text column", choices=cols, selected=cols[0] if cols else None),
                ui.input_select("col_date", "Date column", choices=cols, selected=cols[0] if cols else None),
                ui.input_select("col_title", "Title (optional)", choices=["None"] + cols, selected="None"),
            )
        except: return ui.p("Error reading file", style="color:#ef4444;font-size:.78rem;")

    @reactive.Calc
    def time_labels_all():
        f = input.file()
        if f is None: return []
        try:
            df = pd.read_csv(f[0]["datapath"])
            df["_d"] = pd.to_datetime(df[input.col_date()], errors="coerce")
            df = df.dropna(subset=["_d"])
            periods = df["_d"].dt.to_period(input.granularity()).astype(str)
            return sorted(periods.unique().tolist())
        except: return []

    @output
    @render.ui
    def time_range_ui():
        tl = time_labels_all()
        if not tl: return ui.div()
        n = len(tl)
        return ui.div(
            ui.input_slider("time_range", "Time window", min=0, max=n - 1, value=[0, n - 1], step=1, ticks=False),
            ui.output_ui("time_range_labels"),
            style="margin-bottom:4px;"
        )

    @output
    @render.ui
    def time_range_labels():
        tl = time_labels_all()
        if not tl: return ui.div()
        try:
            r = input.time_range()
            start, end = tl[int(r[0])], tl[int(r[1])]
        except:
            start, end = tl[0], tl[-1]
        
        
        return ui.tags.div(
            ui.tags.span(start, class_="date-badge"),
            ui.tags.span("→", style="color:var(--muted); font-size:0.8rem; display:flex; align-items:center;"),
            ui.tags.span(end, class_="date-badge"),
            style="display:flex; justify-content:space-between; margin-top:8px;"
        )

    @reactive.effect
    @reactive.event(input.time_range, input.granularity)
    def _mark_dirty():
        if filter_state.get() is not None: filter_dirty.set(True)

    def _get_range():
        tl = time_labels_all()
        if not tl: return None, None
        try:
            r = input.time_range()
            return tl[int(r[0])], tl[int(r[1])]
        except: return tl[0], tl[-1]

    @reactive.Calc
    def quick_stats():
        f = input.file()
        if f is None: return None
        tl = time_labels_all()
        t_start, t_end = _get_range()
        if not tl or t_start is None: return None
        try:
            df = pd.read_csv(f[0]["datapath"])
            df["_d"] = pd.to_datetime(df[input.col_date()], errors="coerce")
            df = df.dropna(subset=["_d"])
            df["_per"] = df["_d"].dt.to_period(input.granularity()).astype(str)
            df = df[(df["_per"] >= t_start) & (df["_per"] <= t_end)].copy()
            wc = df[input.col_content()].astype(str).apply(lambda x: len(x.split()))
            return {"n_docs": len(df), "avg_len": float(wc.mean()) if len(wc)>0 else 0.0, "n_slices": len(df["_per"].unique())}
        except: return None

    def _run_filter_pipeline(progress_ctx):
        f = input.file()
        t_start, t_end = _get_range()
        dtm_core.LANGUAGE = input.language()

        corpus, vocab, time_labels_f, beta_init, alpha_init = load_csv_data(
            path=f[0]["datapath"], col_date=input.col_date(), col_text=input.col_content(),
            granularity=input.granularity(), n_topics=input.topics(),
            max_features=100000, min_df=5, max_df=0.75, language=input.language(),
            progress_callback=lambda p, m: progress_ctx.set(int(p*0.9), message=m) if progress_ctx else None,
            date_start=t_start, date_end=t_end,
        )

        df = pd.read_csv(f[0]["datapath"]).dropna(subset=[input.col_date(), input.col_content()])
        df["_d"] = pd.to_datetime(df[input.col_date()], errors="coerce")
        df["_per"] = df["_d"].dt.to_period(input.granularity()).astype(str)
        df = df[(df["_per"] >= t_start) & (df["_per"] <= t_end)].copy()
        wc = df[input.col_content()].astype(str).apply(lambda x: len(x.split()))

        return {
            "corpus": corpus, "vocab": vocab, "time_labels": time_labels_f, "beta_init": beta_init, "alpha_init": alpha_init,
            "df": df, "word_counts": wc, "n_docs": len(df), "vocab_size": len(vocab), "avg_len": float(wc.mean()) if len(wc)>0 else 0.0
        } 

    @reactive.effect
    @reactive.event(input.apply_filter)
    def apply_filter():
        if input.file() is None: return
        loading.set(True)
        prog_pct.set(5)
        prog_msg.set("Filtering corpus…")
        with ui.Progress(min=0, max=100) as p:
            try:
                fs = _run_filter_pipeline(p)
                filter_state.set(fs)
                filter_dirty.set(False)
            except Exception as e:
                prog_msg.set(f"Error: {e}")
                loading.set(False)
                return
        loading.set(False)
        prog_pct.set(100)
        prog_msg.set(f"Filter applied: {fs['n_docs']} docs")

    @reactive.effect
    @reactive.event(input.run)
    def train():
        if input.file() is None: return
        loading.set(True)
        prog_pct.set(5)
        prog_msg.set("Preparing corpus…")

        with ui.Progress(min=0, max=100) as p:
            fs = filter_state.get()
            if fs is None or filter_dirty.get():
                fs = _run_filter_pipeline(p)
                filter_state.set(fs)
                filter_dirty.set(False)

            p.set(45, message="Initialising model…")
            model = DynamicTopicModel(
                num_topics=input.topics(), vocab_size=len(fs["vocab"]), num_times=len(fs["time_labels"]),
                sigma2=0.05, delta2=0.05, beta_init=fs["beta_init"], alpha_init=fs["alpha_init"]
            )
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.02)
            history = []
            epochs = input.epochs()

            for i in range(epochs):
                optimizer.zero_grad()
                loss = model.compute_elbo(fs["corpus"])
                (-loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
                
                ev = loss.item()
                history.append(ev)
                
                pct = 45 + int((i + 1) / epochs * 55)
                msg_epoch = f"Epoch {i+1}/{epochs} - ELBO: {ev:.1f}"
                p.set(pct, message=msg_epoch)
                prog_pct.set(pct)
                prog_msg.set(msg_epoch)

                if i > 5 and abs(history[-1] - history[-2]) / (abs(history[-2]) + 1e-12) < 1e-5:
                    conv_msg.set(f"Convergence reached at epoch {i+1}.")
                    break
            else:
                conv_msg.set(f"Max iterations reached ({epochs} epochs).")

        model_state.set({
            "model": model, "history": history, "vocab": fs["vocab"], "time": fs["time_labels"],
            "corpus": fs["corpus"],
            "df": fs["df"], "word_counts": fs["word_counts"], "lr": optimizer.param_groups[0]["lr"],
            "n_docs": fs["n_docs"], "vocab_size": fs["vocab_size"], "avg_len": fs["avg_len"]
        })

        ui.update_select("topic_id", choices=[str(i) for i in range(input.topics())])
        ui.update_select("time_slice", choices={str(i): str(t) for i, t in enumerate(fs["time_labels"])})
        loading.set(False)
        prog_pct.set(100)
        prog_msg.set("Model trained successfully!")

    @output
    @render.ui
    def status():
        msg = prog_msg.get()
        if not msg: return ui.tags.p("Ready to process.", style="font-size:0.8rem; color:var(--muted);")
        color = "var(--accent)" if loading.get() else ("#16a34a" if "trained" in msg or "applied" in msg else "#ef4444")
        return ui.tags.p(msg, style=f"font-size:0.8rem; font-weight:600; font-family:var(--mono); color:{color};")

    @output
    @render.ui
    def progress_bar():
        pct = prog_pct.get()
        if not loading.get() and pct == 0: return ui.div()
        color = "#16a34a" if pct == 100 else "#0ea5e9"
        return ui.div(
            ui.tags.div(ui.tags.div(style=f"width:{pct}%; background:{color}; height:100%; border-radius:4px; transition:width 0.2s ease;"), class_="dtm-progress-track"),
            ui.tags.span(f"{pct}%" if pct < 100 else "Done", style=f"color:{color}; font-size:0.65rem; font-family:var(--mono); margin-top:4px; display:block;")
        )

    @output
    @render.ui
    def overview_stats():
        ms, fs, qs = model_state.get(), filter_state.get(), quick_stats()
        
        
        if ms: data, status_txt, cls = ms, "● Model Trained", "trained"
        elif fs: data, status_txt, cls = fs, "● Filter Applied", "filtered"
        elif qs: data, status_txt, cls = qs, "○ Filter Pending", "pending"
        else: return ui.p("Upload a CSV file to see data statistics.", style="color:var(--muted);")

        def fmt(v): return "—" if v is None else (f"{v/1e6:.1f}M" if v>=1e6 else (f"{v/1e3:.1f}k" if v>=1e3 else str(v)))

        return ui.div(
            ui.tags.span(status_txt, class_=f"status-badge {cls}"),
            ui.tags.div(
                ui.tags.div(
                    ui.tags.span("Documents", class_="stat-card-label"),
                    ui.tags.div(fmt(data["n_docs"]), class_="stat-card-value"),
                    ui.tags.div("in filtered set", class_="stat-card-sub"),
                    class_="stat-card docs"
                ),
                ui.tags.div(
                    ui.tags.span("Vocabulary", class_="stat-card-label"),
                    ui.tags.div(fmt(data.get("vocab_size")), class_="stat-card-value"),
                    ui.tags.div("unique words", class_="stat-card-sub"),
                    class_="stat-card vocab"
                ),
                ui.tags.div(
                    ui.tags.span("Avg. length", class_="stat-card-label"),
                    ui.tags.div(f"{data['avg_len']:.0f}", class_="stat-card-value"),
                    ui.tags.div("words per doc", class_="stat-card-sub"),
                    class_="stat-card avgl"
                ),
                class_="stats-row"
            )
        )

    @output
    @render_widget
    def corpus():
        ms = model_state.get()
        fs = filter_state.get()
        data = ms if ms else (fs if fs and not filter_dirty.get() else None)
        
        if not data: return _empty_fig("Apply a filter or train a model to get started.")
        df, wc = data["df"], data["word_counts"]
        
        fig = make_subplots(rows=1, cols=2, subplot_titles=["Document length (words)", "Documents per period"], horizontal_spacing=0.1)
        fig.add_trace(go.Histogram(x=wc, marker_color="#0ea5e9", name="Length"), row=1, col=1)
        fig.add_trace(go.Histogram(x=df["_per"].astype(str), marker_color="#8b5cf6", name="Docs"), row=1, col=2)
        
        fig.update_layout(
            template="plotly_white", showlegend=False, height=340,
            margin=dict(l=48, r=40, t=56, b=48),
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif", size=12, color="#475569"),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0"),
            xaxis2=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0"),
            yaxis2=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        )
        return fig

    @output
    @render_widget
    def topics_plot():
        ms = model_state.get()
        if not ms: return _empty_fig()
        model, vocab, topic_id, alpha = ms["model"], ms["vocab"], int(input.topic_id()), float(input.specificity() or 0.0)
        
        m_t, _ = get_smoothed_beta(model, topic_id)
        probs = torch.softmax(m_t, dim=-1).detach().numpy()
        freq = probs.mean(axis=0)

        if alpha > 0:
            all_freq = [torch.softmax(get_smoothed_beta(model, k)[0], dim=-1).detach().numpy().mean(axis=0) for k in range(model.K)]
            mean_across = np.clip(np.mean(all_freq, axis=0), 1e-10, None)
            lift = freq / mean_across
            score = (freq ** (1 - alpha)) * (lift ** alpha)
        else:
            score = freq

        top = np.argsort(score)[-int(input.n_words_plot() or 8):][::-1]
        x = [str(t) for t in ms["time"]]
        sw = _adaptive_smooth_window(len(x))

        fig = go.Figure()
        for idx, w in enumerate(top):
            fig.add_trace(go.Scatter(x=x, y=_smooth(probs[:, w], sw), mode="lines", name=vocab[w], line=dict(color=COLORS_20[idx % len(COLORS_20)], width=2.5)))
        fig.update_layout(
            template="plotly_white", hovermode="x unified", height=440,
            margin=dict(t=24, b=40, l=48, r=24),
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif", size=12, color="#475569"),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", gridwidth=1, linecolor="#e2e8f0"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", gridwidth=1, linecolor="#e2e8f0"),
            legend=dict(font=dict(size=11), bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1),
        )
        return fig

    @output
    @render_widget
    def top_words_time():
        ms = model_state.get()
        if not ms: return _empty_fig()
        try: t_idx = int(input.time_slice())
        except: t_idx = len(ms["time"]) - 1
        
        model, vocab, K = ms["model"], ms["vocab"], ms["model"].K
        alpha = float(input.specificity() or 0.0)
        n_words = int(input.n_words_plot() or 8)
        
        if alpha > 0:
            all_freq_t = [torch.softmax(get_smoothed_beta(model, k)[0][t_idx], dim=-1).detach().numpy() for k in range(K)]
            mean_across = np.clip(np.mean(all_freq_t, axis=0), 1e-10, None)

        topic_rows = int(np.ceil(K / 2))
        total_rows = topic_rows + 1
        subplot_titles = [f"Topic {k}" for k in range(K)] + [""] * (topic_rows * 2 - K) + ["All topics"]
        specs = [[{}, {}]] * topic_rows + [[{"colspan": 2}, None]]

        fig = make_subplots(
            rows=total_rows, cols=2,
            subplot_titles=subplot_titles,
            specs=specs,
            vertical_spacing=0.08,
        )

        for k in range(K):
            probs = torch.softmax(get_smoothed_beta(model, k)[0][t_idx], dim=-1).detach().numpy()
            score = (probs ** (1 - alpha)) * ((probs / mean_across) ** alpha) if alpha > 0 else probs
            top = np.argsort(score)[-n_words:][::-1]
            fig.add_trace(go.Bar(x=[vocab[i] for i in top], y=probs[top], marker_color=COLORS_10[k % len(COLORS_10)]), row=(k // 2) + 1, col=(k % 2) + 1)

        # --- Top mots globaux (somme sur les topics) à la période courante ---
        corpus_counts = ms["corpus"]                              # [T, K, V]
        global_counts = corpus_counts[t_idx].sum(dim=0).numpy()  # [V]
        top_global = np.argsort(global_counts)[-n_words:][::-1]
        fig.add_trace(
            go.Bar(
                x=[vocab[i] for i in top_global],
                y=global_counts[top_global],
                marker_color="#94a3b8",
            ),
            row=total_rows, col=1,
        )

        fig.update_layout(
            template="plotly_white", showlegend=False,
            height=260 * topic_rows + 220,
            margin=dict(t=56, b=48, l=48, r=24),
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif", size=11, color="#475569"),
        )
        return fig

    @output
    @render_widget
    def evolution():
        ms = model_state.get()
        if not ms: return _empty_fig()
        props, _ = get_smoothed_alpha(ms["model"])
        x = [str(t) for t in ms["time"]]
        sw = _adaptive_smooth_window(len(x))
        
        
        fig = go.Figure()
        for k in range(props.shape[1]):
            fig.add_trace(go.Scatter(
                x=x, 
                y=_smooth(props[:, k], sw), 
                mode="lines", 
                name=f"Topic {k}", 
                line=dict(color=COLORS_10[k % len(COLORS_10)], width=2.5)
            ))
        fig.update_layout(
            template="plotly_white", hovermode="x unified", height=440,
            margin=dict(t=24, b=40, l=48, r=24),
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif", size=12, color="#475569"),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", gridwidth=1, linecolor="#e2e8f0"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", gridwidth=1, linecolor="#e2e8f0"),
            legend=dict(font=dict(size=11), bgcolor="rgba(255,255,255,0.8)", bordercolor="#e2e8f0", borderwidth=1),
        )
        return fig

    @output
    @render_widget
    def elbo():
        ms = model_state.get()
        if not ms: return _empty_fig("Train a model to see the loss curve.")
        fig = go.Figure(go.Scatter(y=ms["history"], mode="lines", line=dict(color="#0ea5e9", width=2.5), fill="tozeroy", fillcolor="rgba(14,165,233,0.06)"))
        fig.update_layout(
            template="plotly_white", xaxis_title="Epoch", yaxis_title="ELBO", height=300,
            margin=dict(t=24, b=48, l=56, r=24),
            font=dict(family="-apple-system, BlinkMacSystemFont, 'Inter', sans-serif", size=12, color="#475569"),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0"),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9", linecolor="#e2e8f0"),
        )
        return fig

    @output
    @render.ui
    def convergence_status():
        msg = conv_msg.get()
        if loading.get(): return ui.div(prog_msg.get(), style="color:var(--accent); font-weight:600; padding:10px; background:#e0f2fe; border-radius:10px;")
        if not msg: return ui.p("No training run yet.", style="color:var(--muted);")
        color = "#16a34a" if "reached" in msg or "atteinte" in msg else "#d97706"
        return ui.div(msg, style=f"color:{color}; font-weight:600; padding:10px; background:{color}1A; border-radius:10px;")

    @output
    @render.ui
    def lr_display():
        ms = model_state.get()
        if not ms: return ui.p("—", style="color:var(--muted); font-family:var(--mono);")
        return ui.div(f"{ms['lr']:.5f}", style="font-family:var(--mono); font-size:1.4rem; font-weight:700; color:#0ea5e9;")


    @output
    @render.ui
    def wordclouds():
        ms = model_state.get()
        if not ms: return ui.p("Train a model to see word clouds.", style="color:var(--muted);")
        model, vocab, K = ms["model"], ms["vocab"], ms["model"].K
        imgs = []
        for k in range(K):
            m_t, _ = get_smoothed_beta(model, k)
            probs = torch.softmax(m_t.mean(dim=0), dim=-1).detach().numpy()
            freq = {vocab[i]: float(probs[i]) for i in range(len(vocab))}
            color = COLORS_10[k % len(COLORS_10)]
            def make_color_func(c):
                import random
                r, g, b = int(c[1:3],16), int(c[3:5],16), int(c[5:7],16)
                def _f(*args, **kwargs):
                    lum = random.randint(0, 60)
                    return f"rgb({min(r+lum,255)},{min(g+lum,255)},{min(b+lum,255)})"
                return _f
            wc = WordCloud(
                width=400, height=220, background_color="white",
                max_words=60, color_func=make_color_func(color),
                prefer_horizontal=0.9, margin=4,
            ).generate_from_frequencies(freq)
            buf = io.BytesIO()
            wc.to_image().save(buf, format="PNG")
            b64 = __import__("base64").b64encode(buf.getvalue()).decode()
            imgs.append(ui.tags.div(
                ui.tags.p(f"Topic {k}", style=f"font-weight:700; color:{color}; margin:0 0 6px 0; font-size:0.85rem; text-transform:uppercase; letter-spacing:.05em;"),
                ui.tags.img(src=f"data:image/png;base64,{b64}", style="width:100%; border-radius:8px;"),
                style="flex:1; min-width:280px; max-width:420px;"
            ))
        return ui.tags.div(*imgs, style="display:flex; flex-wrap:wrap; gap:16px;")

app = App(app_ui, server)