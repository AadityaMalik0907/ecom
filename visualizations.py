

import calendar
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import pandas as pd
import numpy as np
from itertools import combinations
from collections import Counter

# ─────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────

PALETTE = [
    "#4ECDC4", "#FF6B6B", "#FFE66D", "#A78BFA",
    "#F97316", "#06B6D4", "#10B981", "#F43F5E",
    "#8B5CF6", "#FBBF24", "#34D399", "#60A5FA",
]

BG_DARK   = "#0F172A"   # slide background
BG_CARD   = "#1E293B"   # panel background
GRID_CLR  = "#334155"   # subtle grid
TEXT_CLR  = "#F1F5F9"   # primary text
SUB_CLR   = "#94A3B8"   # secondary text
ACCENT    = "#4ECDC4"   # teal accent
ACCENT2   = "#FF6B6B"   # coral accent

def _apply_dark_style(fig, axes_list=None):
    """Apply a consistent dark, clean style to a figure."""
    fig.patch.set_facecolor(BG_DARK)
    if axes_list is None:
        axes_list = fig.get_axes()
    for ax in axes_list:
        ax.set_facecolor(BG_CARD)
        ax.tick_params(colors=SUB_CLR, labelsize=9)
        ax.xaxis.label.set_color(SUB_CLR)
        ax.yaxis.label.set_color(SUB_CLR)
        ax.title.set_color(TEXT_CLR)
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID_CLR)
        ax.grid(color=GRID_CLR, linewidth=0.5, alpha=0.6)

def _title(fig, text, subtitle=""):
    y = 0.97 if subtitle else 0.96
    fig.text(0.5, y, text, ha="center", va="top",fontsize=16, fontweight="bold", color=TEXT_CLR,fontfamily="monospace")
    if subtitle:
        fig.text(0.5, y - 0.035, subtitle, ha="center", va="top",fontsize=10, color=SUB_CLR)

def _bar(ax, x, y, colors=None, horizontal=False, **kwargs):
    """Gradient-coloured bar helper."""
    if colors is None:
        colors = [PALETTE[i % len(PALETTE)] for i in range(len(x))]
    if horizontal:
        bars = ax.barh(x, y, color=colors, edgecolor="none", **kwargs)
    else:
        bars = ax.bar(x, y, color=colors, edgecolor="none", **kwargs)
    return bars

def _donut(ax, sizes, labels, colors=None, center_text=""):
    if colors is None:
        colors = PALETTE[: len(sizes)]
    wedges, texts, autotexts = ax.pie(
        sizes, labels=None, autopct="%1.1f%%",
        startangle=90, colors=colors,
        wedgeprops=dict(width=0.55, edgecolor=BG_DARK, linewidth=2),
        pctdistance=0.78,
    )
    for at in autotexts:
        at.set_color(BG_DARK)
        at.set_fontsize(8)
        at.set_fontweight("bold")
    ax.legend(wedges, labels, loc="center left",bbox_to_anchor=(0.85, 0.5), fontsize=8,frameon=False, labelcolor=TEXT_CLR)
    if center_text:
        ax.text(0, 0, center_text, ha="center", va="center",
                fontsize=11, color=TEXT_CLR, fontweight="bold")
    ax.set_facecolor(BG_CARD)


# ─────────────────────────────────────────────
# 1. MONTHLY SALES — histogram + trend line
#    (matches the uploaded reference image style)
# ─────────────────────────────────────────────

def plot_monthly_sales(df: pd.DataFrame) -> plt.Figure:
    """Bar + trend-line chart of monthly revenue (ref: uploaded image)."""
    sales = df.groupby("Month")["TotalPrice"].sum().reset_index()
    month_names = [calendar.month_abbr[int(m)] for m in sales["Month"]]
    values = sales["TotalPrice"].values

    fig, ax = plt.subplots(figsize=(15, 7))
    _apply_dark_style(fig, [ax])

    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(month_names))]
    ax.bar(month_names, values, color=bar_colors, alpha=0.85, edgecolor=BG_DARK, linewidth=0.8)
    ax.plot(month_names, values, color=ACCENT2, linewidth=3,
            marker="o", markersize=9, markerfacecolor=BG_DARK,
            markeredgecolor=ACCENT2, markeredgewidth=2, zorder=5, label="Sales Trend")

    for i, (mn, v) in enumerate(zip(month_names, values)):
        ax.text(i, v + max(values) * 0.01, f"{v/1e3:.0f}K",
                ha="center", fontsize=8, color=TEXT_CLR, fontweight="bold")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax.set_xlabel("Month", fontsize=11, labelpad=8)
    ax.set_ylabel("Total Revenue (£)", fontsize=11, labelpad=8)
    ax.legend(facecolor=BG_CARD, edgecolor=GRID_CLR, labelcolor=TEXT_CLR, fontsize=10)
    _title(fig, "Monthly Sales Analysis", "Revenue bars with trend line overlay")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────
# 2. COUNTRY REVENUE — donut + horizontal bar
# ─────────────────────────────────────────────

def plot_country_analysis(df: pd.DataFrame) -> plt.Figure:
    country_rev   = df.groupby("Country")["TotalPrice"].sum().sort_values(ascending=False)
    country_orders= df.groupby("Country")["InvoiceNo"].nunique().sort_values(ascending=False)

    top5   = country_rev.head(5)
    other  = country_rev.iloc[5:].sum()
    pie_labels = top5.index.tolist() + ["Others"]
    pie_sizes  = top5.values.tolist() + [other]

    top10_rev   = country_rev.head(10)
    top10_ord   = country_orders.head(10)

    fig = plt.figure(figsize=(18, 9))
    _apply_dark_style(fig)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Donut – revenue share
    ax1 = fig.add_subplot(gs[0, 0])
    _donut(ax1, pie_sizes, pie_labels,center_text=f"£{country_rev.sum()/1e6:.1f}M")
    ax1.set_title("Revenue Share by Country", color=TEXT_CLR, fontsize=12, pad=10)

    # Horizontal bar – top 10 revenue
    ax2 = fig.add_subplot(gs[0, 1])
    _apply_dark_style(fig, [ax2])
    _bar(ax2, top10_rev.index[::-1], top10_rev.values[::-1],colors=[PALETTE[i % len(PALETTE)] for i in range(10)], horizontal=True)
    ax2.set_xlabel("Total Revenue (£)", fontsize=10)
    ax2.set_title("Top 10 Countries by Revenue", color=TEXT_CLR, fontsize=12)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

    # Horizontal bar – top 10 orders
    ax3 = fig.add_subplot(gs[0, 2])
    _apply_dark_style(fig, [ax3])
    _bar(ax3, top10_ord.index[::-1], top10_ord.values[::-1],colors=[PALETTE[(i + 3) % len(PALETTE)] for i in range(10)], horizontal=True)
    ax3.set_xlabel("Unique Orders", fontsize=10)
    ax3.set_title("Top 10 Countries by Orders", color=TEXT_CLR, fontsize=12)

    _title(fig, "Country-Level Analysis", "Revenue share · Top revenue · Top order volume")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────
# 3. TOP PRODUCTS
# ─────────────────────────────────────────────

def plot_top_products(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    product_sales = (
        df.groupby("Description")
        .agg(TotalRevenue=("TotalPrice", "sum"), TotalQty=("Quantity", "sum"))
        .sort_values("TotalRevenue", ascending=False)
    )
    top  = product_sales.head(top_n)
    low  = product_sales.tail(10)

    fig = plt.figure(figsize=(18, 10))
    _apply_dark_style(fig)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Top N by revenue
    ax1 = fig.add_subplot(gs[0, :])
    _apply_dark_style(fig, [ax1])
    labels = [d[:30] + "…" if len(d) > 30 else d for d in top.index]
    _bar(ax1, labels, top["TotalRevenue"],colors=[PALETTE[i % len(PALETTE)] for i in range(len(top))])
    ax1.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
    ax1.set_title(f"Top {top_n} Products by Revenue", color=TEXT_CLR, fontsize=13)

    # Revenue distribution
    ax2 = fig.add_subplot(gs[1, 0])
    _apply_dark_style(fig, [ax2])
    ax2.hist(product_sales["TotalRevenue"], bins=60, color=ACCENT, alpha=0.8, edgecolor="none")
    ax2.set_title("Revenue Distribution (All Products)", color=TEXT_CLR, fontsize=11)
    ax2.set_xlabel("Revenue (£)", fontsize=9)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

    # Qty distribution
    ax3 = fig.add_subplot(gs[1, 1])
    _apply_dark_style(fig, [ax3])
    ax3.hist(product_sales["TotalQty"], bins=60, color=ACCENT2, alpha=0.8, edgecolor="none")
    ax3.set_title("Quantity Distribution (All Products)", color=TEXT_CLR, fontsize=11)
    ax3.set_xlabel("Units Sold", fontsize=9)

    _title(fig, "Product Performance", f"Top {top_n} revenue leaders & distribution tails")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────
# 4. RFM SEGMENT OVERVIEW
# ─────────────────────────────────────────────

def plot_rfm_overview(rfm: pd.DataFrame) -> plt.Figure:
    seg_cnt = rfm["Segment"].value_counts()
    seg_rev = rfm.groupby("Segment")["Monetary"].sum().reindex(seg_cnt.index)
    seg_colors = [PALETTE[i % len(PALETTE)] for i in range(len(seg_cnt))]

    fig = plt.figure(figsize=(18, 10))
    _apply_dark_style(fig)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    # Customer count bar
    ax1 = fig.add_subplot(gs[0, 0:2])
    _apply_dark_style(fig, [ax1])
    _bar(ax1, seg_cnt.index, seg_cnt.values, colors=seg_colors)
    for i, v in enumerate(seg_cnt.values):
        ax1.text(i, v + seg_cnt.max() * 0.02, str(v),ha="center", fontsize=9, color=TEXT_CLR, fontweight="bold")
    ax1.set_title("Customer Count by Segment", color=TEXT_CLR, fontsize=12)
    ax1.set_xticklabels(seg_cnt.index, rotation=25, ha="right", fontsize=9)

    # Donut – segment revenue
    ax2 = fig.add_subplot(gs[0, 2])
    _donut(ax2, seg_rev.values, seg_rev.index, colors=seg_colors,center_text="Revenue\nShare")
    ax2.set_title("Revenue by Segment", color=TEXT_CLR, fontsize=12, pad=10)

    # Revenue total bar
    ax3 = fig.add_subplot(gs[1, 0:2])
    _apply_dark_style(fig, [ax3])
    seg_rev_sorted = seg_rev.sort_values(ascending=False)
    _bar(ax3, seg_rev_sorted.index, seg_rev_sorted.values,colors=[PALETTE[i % len(PALETTE)] for i in range(len(seg_rev_sorted))])
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
    ax3.set_title("Total Revenue by Segment", color=TEXT_CLR, fontsize=12)
    ax3.set_xticklabels(seg_rev_sorted.index, rotation=25, ha="right", fontsize=9)

    # RFM scatter
    ax4 = fig.add_subplot(gs[1, 2])
    _apply_dark_style(fig, [ax4])
    seg_list = rfm["Segment"].unique().tolist()
    for i, seg in enumerate(seg_list):
        mask = rfm["Segment"] == seg
        ax4.scatter(rfm.loc[mask, "Recency"], rfm.loc[mask, "Monetary"],
                    color=PALETTE[i % len(PALETTE)], alpha=0.5, s=18, label=seg)
    ax4.set_xlabel("Recency (days)", fontsize=9)
    ax4.set_ylabel("Monetary (£)", fontsize=9)
    ax4.set_title("Recency vs Monetary", color=TEXT_CLR, fontsize=12)
    ax4.legend(fontsize=7, facecolor=BG_CARD, edgecolor=GRID_CLR,labelcolor=TEXT_CLR, markerscale=1.5)
    ax4.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))

    _title(fig, "RFM Customer Segmentation", "Count · Revenue · Geographic scatter")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────
# 5. RFM HEATMAP
# ─────────────────────────────────────────────

def plot_rfm_heatmap(rfm: pd.DataFrame) -> plt.Figure:
    pivot = rfm.pivot_table(index="R", columns="F", values="Monetary", aggfunc="mean")
    corr  = rfm[["Recency", "Frequency", "Monetary"]].corr()

    cmap_teal = LinearSegmentedColormap.from_list(
        "teal_dark", [BG_CARD, ACCENT], N=256
    )

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    _apply_dark_style(fig, axes)

    sns.heatmap(pivot, annot=True, fmt=".0f", cmap=cmap_teal,
                linewidths=0.5, linecolor=BG_DARK,
                cbar_kws={"shrink": 0.8}, ax=axes[0])
    axes[0].set_title("Avg Monetary Value — R × F Matrix", color=TEXT_CLR, fontsize=12, pad=10)
    axes[0].tick_params(colors=SUB_CLR)

    cmap_corr = LinearSegmentedColormap.from_list(
        "corr", [ACCENT2, BG_CARD, ACCENT], N=256
    )
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap_corr,
                linewidths=0.5, linecolor=BG_DARK,
                vmin=-1, vmax=1, square=True, ax=axes[1])
    axes[1].set_title("RFM Correlation Matrix", color=TEXT_CLR, fontsize=12, pad=10)
    axes[1].tick_params(colors=SUB_CLR)

    _title(fig, "RFM Heatmaps", "R×F average monetary · Correlation between R, F, M")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ─────────────────────────────────────────────
# 6. SEGMENT BEHAVIOUR
# ─────────────────────────────────────────────

def plot_segment_behaviour(df: pd.DataFrame, rfm: pd.DataFrame) -> plt.Figure:
    df_rfm = df.merge(rfm[["Segment"]], left_on="CustomerID", right_index=True)
    sb = df_rfm.groupby("Segment").agg(
        Orders=("InvoiceNo", "nunique"),
        Customers=("CustomerID", "nunique"),
        Revenue=("TotalPrice", "sum"),
        Quantity=("Quantity", "sum"),
    )
    sb["AOV"]               = sb["Revenue"] / sb["Orders"]
    sb["Purchase_Frequency"]= sb["Orders"]  / sb["Customers"]
    seg_colors = [PALETTE[i % len(PALETTE)] for i in range(len(sb))]

    fig = plt.figure(figsize=(18, 9))
    _apply_dark_style(fig)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    for idx, (col, title, fmt) in enumerate([
        ("AOV",               "Avg Order Value by Segment",    "£{x:,.0f}"),
        ("Purchase_Frequency","Purchase Frequency by Segment",  "{x:.1f}"),
        ("Revenue",           "Total Revenue by Segment",       "£{x/1e3:.0f}K"),
    ]):
        ax = fig.add_subplot(gs[0, idx])
        _apply_dark_style(fig, [ax])
        _bar(ax, sb.index, sb[col], colors=seg_colors)
        ax.set_xticklabels(sb.index, rotation=30, ha="right", fontsize=8)
        ax.set_title(title, color=TEXT_CLR, fontsize=11)
        ax.yaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _, f=fmt: eval(f'f"{f}"'))
        )

    _title(fig, "Segment Behaviour Deep-Dive", "AOV · Purchase frequency · Total revenue per segment")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ─────────────────────────────────────────────
# 7. ORDER VALUE DISTRIBUTION
# ─────────────────────────────────────────────

def plot_order_value_distribution(df: pd.DataFrame) -> plt.Figure:
    ov  = df.groupby("InvoiceNo")["TotalPrice"].sum()
    items = df.groupby("InvoiceNo")["Quantity"].sum()

    def cat(v):
        if v <= 50:   return "Micro (<£50)"
        if v <= 150:  return "Small (£50-150)"
        if v <= 500:  return "Medium (£150-500)"
        return "Large (£500+)"

    ov_df = ov.rename("OrderValue").reset_index()
    ov_df["Category"] = ov_df["OrderValue"].apply(cat)
    cat_cnt = ov_df["Category"].value_counts().reindex(
        ["Micro (<£50)", "Small (£50-150)", "Medium (£150-500)", "Large (£500+)"]
    ).dropna()

    fig = plt.figure(figsize=(18, 9))
    _apply_dark_style(fig)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    # Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    _apply_dark_style(fig, [ax1])
    clipped = ov[ov < ov.quantile(0.98)]
    ax1.hist(clipped, bins=60, color=ACCENT, alpha=0.85, edgecolor="none")
    ax1.axvline(ov.mean(), color=ACCENT2, linestyle="--", linewidth=2,
                label=f"Mean £{ov.mean():.0f}")
    ax1.axvline(ov.median(), color=PALETTE[2], linestyle="--", linewidth=2,
                label=f"Median £{ov.median():.0f}")
    ax1.legend(fontsize=8, facecolor=BG_CARD, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    ax1.set_xlabel("Order Value (£)", fontsize=9)
    ax1.set_title("Order Value Distribution", color=TEXT_CLR, fontsize=12)

    # Category donut
    ax2 = fig.add_subplot(gs[0, 1])
    cat_colors = [PALETTE[i] for i in range(len(cat_cnt))]
    _donut(ax2, cat_cnt.values, cat_cnt.index, colors=cat_colors,center_text=f"{len(ov):,}\nOrders")
    ax2.set_title("Orders by Value Tier", color=TEXT_CLR, fontsize=12, pad=10)

    # Box plot
    ax3 = fig.add_subplot(gs[0, 2])
    _apply_dark_style(fig, [ax3])
    bp = ax3.boxplot(clipped, vert=True, patch_artist=True,boxprops=dict(facecolor=ACCENT, alpha=0.6),medianprops=dict(color=ACCENT2, linewidth=2.5),whiskerprops=dict(color=SUB_CLR),capprops=dict(color=SUB_CLR),flierprops=dict(marker=".", color=SUB_CLR, markersize=3, alpha=0.4))
    ax3.set_title("Order Value Box Plot", color=TEXT_CLR, fontsize=12)
    ax3.set_ylabel("Order Value (£)", fontsize=9)
    ax3.set_xticks([])

    _title(fig, "Order Value Analysis", "Distribution · Value tiers · Box plot spread")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ─────────────────────────────────────────────
# 8. BASKET SIZE ANALYSIS
# ─────────────────────────────────────────────

def plot_basket_size(df: pd.DataFrame) -> plt.Figure:
    items_df = (
        df.groupby("InvoiceNo")["Quantity"]
        .sum()
        .reset_index()
        .rename(columns={"Quantity": "TotalItems"})
    )

    def cat(v):
        if v <= 10:  return "Small (1-10)"
        if v <= 25:  return "Medium (11-25)"
        if v <= 50:  return "Large (26-50)"
        return "XL (50+)"

    items_df["SizeCat"] = items_df["TotalItems"].apply(cat)
    cat_order = ["Small (1-10)", "Medium (11-25)", "Large (26-50)", "XL (50+)"]
    cat_cnt   = items_df["SizeCat"].value_counts().reindex(cat_order).dropna()

    sorted_items  = items_df["TotalItems"].sort_values().reset_index(drop=True)
    cumulative_pct= (sorted_items.index + 1) / len(sorted_items) * 100

    fig = plt.figure(figsize=(18, 9))
    _apply_dark_style(fig)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    # Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    _apply_dark_style(fig, [ax1])
    ax1.hist(items_df["TotalItems"], bins=40, color=PALETTE[1], alpha=0.8, edgecolor="none")
    ax1.axvline(items_df["TotalItems"].mean(), color=ACCENT2, linestyle="--",
                linewidth=2, label=f"Mean: {items_df['TotalItems'].mean():.1f}")
    ax1.axvline(items_df["TotalItems"].median(), color=PALETTE[2], linestyle="--",
                linewidth=2, label=f"Median: {items_df['TotalItems'].median():.0f}")
    ax1.legend(fontsize=8, facecolor=BG_CARD, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)
    ax1.set_xlabel("Items per Order", fontsize=9)
    ax1.set_title("Distribution of Basket Sizes", color=TEXT_CLR, fontsize=12)

    # Donut – size categories
    ax2 = fig.add_subplot(gs[0, 1])
    _donut(ax2, cat_cnt.values, cat_cnt.index,colors=[PALETTE[i] for i in range(len(cat_cnt))],center_text="Basket\nTiers")
    ax2.set_title("Order Size Categories", color=TEXT_CLR, fontsize=12, pad=10)

    # Cumulative distribution
    ax3 = fig.add_subplot(gs[0, 2])
    _apply_dark_style(fig, [ax3])
    ax3.plot(sorted_items.values, cumulative_pct, color=ACCENT, linewidth=2.5)
    ax3.axhline(50, color=ACCENT2, linestyle="--", linewidth=1.5, alpha=0.7, label="50th pct")
    ax3.axhline(80, color=PALETTE[2], linestyle="--", linewidth=1.5, alpha=0.7, label="80th pct")
    ax3.fill_between(sorted_items.values, cumulative_pct, alpha=0.15, color=ACCENT)
    ax3.set_xlabel("Items per Order", fontsize=9)
    ax3.set_ylabel("Cumulative %", fontsize=9)
    ax3.set_title("Cumulative Distribution", color=TEXT_CLR, fontsize=12)
    ax3.legend(fontsize=8, facecolor=BG_CARD, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    _title(fig, "Basket Size Analysis", "Item distribution · Size tiers · Cumulative curve")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ─────────────────────────────────────────────
# 9. TIME PATTERNS — day-of-week & hourly heat
# ─────────────────────────────────────────────

def plot_time_patterns(df: pd.DataFrame) -> plt.Figure:
    df = df.copy()
    df["DayName"] = df["InvoiceDate"].dt.day_name()
    df["Hour"]    = df["InvoiceDate"].dt.hour
    df["Month"]   = df["InvoiceDate"].dt.month

    day_order  = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    daily_rev  = df.groupby("DayName")["TotalPrice"].sum().reindex(day_order)
    hourly_rev = df.groupby("Hour")["TotalPrice"].sum()
    monthly_rev= df.groupby("Month")["TotalPrice"].sum()
    month_names= [calendar.month_abbr[m] for m in monthly_rev.index]

    # Heatmap: day × hour
    heat_df = (
        df.groupby(["DayName","Hour"])["TotalPrice"].sum()
        .unstack(fill_value=0)
        .reindex(day_order)
    )

    fig = plt.figure(figsize=(18, 11))
    _apply_dark_style(fig)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.5, wspace=0.4)

    # Monthly bar + trend
    ax1 = fig.add_subplot(gs[0, :])
    _apply_dark_style(fig, [ax1])
    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(month_names))]
    ax1.bar(month_names, monthly_rev.values, color=bar_colors, alpha=0.8, edgecolor="none")
    ax1.plot(month_names, monthly_rev.values, color=ACCENT2, linewidth=3,marker="o", markersize=8, markerfacecolor=BG_DARK,markeredgecolor=ACCENT2, markeredgewidth=2, zorder=5)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
    ax1.set_title("Monthly Revenue Trend", color=TEXT_CLR, fontsize=13)

    # Day-of-week
    ax2 = fig.add_subplot(gs[1, 0])
    _apply_dark_style(fig, [ax2])
    day_colors = [PALETTE[i % len(PALETTE)] for i in range(7)]
    _bar(ax2, daily_rev.index, daily_rev.values, colors=day_colors)
    ax2.set_xticklabels(daily_rev.index, rotation=30, ha="right", fontsize=8)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
    ax2.set_title("Revenue by Day of Week", color=TEXT_CLR, fontsize=11)

    # Hourly
    ax3 = fig.add_subplot(gs[1, 1])
    _apply_dark_style(fig, [ax3])
    ax3.plot(hourly_rev.index, hourly_rev.values, color=ACCENT, linewidth=2.5,marker="o", markersize=5, markerfacecolor=BG_DARK,markeredgecolor=ACCENT, markeredgewidth=1.5)
    ax3.fill_between(hourly_rev.index, hourly_rev.values, alpha=0.2, color=ACCENT)
    ax3.set_xlabel("Hour of Day", fontsize=9)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
    ax3.set_title("Revenue by Hour of Day", color=TEXT_CLR, fontsize=11)

    # Day × Hour heatmap
    ax4 = fig.add_subplot(gs[1, 2])
    cmap_heat = LinearSegmentedColormap.from_list("heat", [BG_CARD, ACCENT2], N=256)
    sns.heatmap(heat_df / 1000, cmap=cmap_heat,
                linewidths=0.2, linecolor=BG_DARK,
                cbar_kws={"label": "Revenue (£K)", "shrink": 0.8}, ax=ax4)
    ax4.set_title("Revenue Heatmap (Day × Hour)", color=TEXT_CLR, fontsize=11)
    ax4.tick_params(colors=SUB_CLR, labelsize=7)

    _title(fig, "Time & Seasonality Patterns", "Monthly · Day-of-week · Hourly · Day×Hour heatmap")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────
# 10. CUSTOMER RETENTION & CHURN
# ─────────────────────────────────────────────

def plot_retention_churn(df: pd.DataFrame, rfm: pd.DataFrame) -> plt.Figure:
    customer_orders = df.groupby("CustomerID")["InvoiceNo"].nunique()
    repeat   = (customer_orders > 1).sum()
    one_time = (customer_orders == 1).sum()
    retention_rate = repeat / len(customer_orders)

    rfm = rfm.copy()
    rfm["Churn"] = pd.cut(
        rfm["Recency"],
        bins=[-1, 30, 90, 180, 365, 10_000],
        labels=["Active", "Warm", "Cooling", "At Risk", "Churned"],
    )
    churn_dist = rfm["Churn"].value_counts().reindex(
        ["Active", "Warm", "Cooling", "At Risk", "Churned"]
    ).dropna()

    fig = plt.figure(figsize=(18, 8))
    _apply_dark_style(fig)
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

    # Repeat vs one-time donut
    ax1 = fig.add_subplot(gs[0, 0])
    _donut(ax1, [repeat, one_time], ["Repeat", "One-time"],colors=[ACCENT, ACCENT2],center_text=f"{retention_rate:.0%}\nRetention")
    ax1.set_title("Repeat vs One-time Customers", color=TEXT_CLR, fontsize=12, pad=10)

    # Churn bucket bar
    ax2 = fig.add_subplot(gs[0, 1])
    _apply_dark_style(fig, [ax2])
    churn_colors = [PALETTE[i] for i in range(len(churn_dist))]
    _bar(ax2, churn_dist.index, churn_dist.values, colors=churn_colors)
    for i, v in enumerate(churn_dist.values):
        ax2.text(i, v + churn_dist.max() * 0.02, str(v),ha="center", fontsize=9, color=TEXT_CLR, fontweight="bold")
    ax2.set_title("Customer Churn Distribution", color=TEXT_CLR, fontsize=12)
    ax2.set_xticklabels(churn_dist.index, rotation=20, ha="right", fontsize=9)

    # Recency histogram coloured by churn zone
    ax3 = fig.add_subplot(gs[0, 2])
    _apply_dark_style(fig, [ax3])
    bins = [0, 30, 90, 180, 365, rfm["Recency"].max() + 1]
    zone_colors = [PALETTE[i] for i in range(5)]
    zone_labels = ["Active (0-30)", "Warm (30-90)", "Cooling (90-180)","At Risk (180-365)", "Churned (365+)"]
    for lo, hi, c, lab in zip(bins[:-1], bins[1:], zone_colors, zone_labels):
        subset = rfm[(rfm["Recency"] >= lo) & (rfm["Recency"] < hi)]["Recency"]
        ax3.hist(subset, bins=30, color=c, alpha=0.8, edgecolor="none", label=lab)
    ax3.set_xlabel("Recency (days)", fontsize=9)
    ax3.set_ylabel("Customer Count", fontsize=9)
    ax3.set_title("Recency Distribution by Zone", color=TEXT_CLR, fontsize=12)
    ax3.legend(fontsize=7, facecolor=BG_CARD, edgecolor=GRID_CLR, labelcolor=TEXT_CLR)

    _title(fig, "Customer Retention & Churn",f"Overall retention rate: {retention_rate:.1%}")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ─────────────────────────────────────────────
# 11. KPI DASHBOARD (EXTRA)
# ─────────────────────────────────────────────

def plot_kpi_dashboard(df: pd.DataFrame, rfm: pd.DataFrame) -> plt.Figure:
    total_revenue  = df["TotalPrice"].sum()
    total_orders   = df["InvoiceNo"].nunique()
    total_customers= df["CustomerID"].nunique()
    aov            = total_revenue / total_orders
    avg_items      = df.groupby("InvoiceNo")["Quantity"].sum().mean()
    top_country    = df.groupby("Country")["TotalPrice"].sum().idxmax()
    champions_pct  = (rfm["Segment"] == "Champions").mean() * 100

    kpis = [
        ("Total Revenue",     f"£{total_revenue/1e6:.2f}M", ACCENT),
        ("Total Orders",      f"{total_orders:,}",           PALETTE[1]),
        ("Unique Customers",  f"{total_customers:,}",        PALETTE[2]),
        ("Avg Order Value",   f"£{aov:.2f}",                 PALETTE[3]),
        ("Avg Basket Size",   f"{avg_items:.1f} items",      PALETTE[4]),
        ("Top Market",        top_country,                   PALETTE[5]),
        ("Champions",         f"{champions_pct:.1f}%",       PALETTE[6]),
    ]

    fig = plt.figure(figsize=(18, 4.5))
    fig.patch.set_facecolor(BG_DARK)
    n = len(kpis)
    for i, (label, value, color) in enumerate(kpis):
        ax = fig.add_axes([i / n + 0.005, 0.15, 0.99 / n - 0.01, 0.65])
        ax.set_facecolor(BG_CARD)
        for spine in ax.spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.5, 0.62, value, ha="center", va="center",
                fontsize=17, fontweight="bold", color=color,
                transform=ax.transAxes, fontfamily="monospace")
        ax.text(0.5, 0.22, label, ha="center", va="center",
                fontsize=9, color=SUB_CLR, transform=ax.transAxes)

    _title(fig, "Key Performance Indicators", "E-Commerce snapshot")
    return fig


# ─────────────────────────────────────────────
# 12. COHORT RETENTION HEATMAP (EXTRA)
# ─────────────────────────────────────────────

def plot_cohort_retention(retention_pct: pd.DataFrame) -> plt.Figure:
    cmap = LinearSegmentedColormap.from_list("cohort", [BG_CARD, ACCENT], N=256)

    fig, ax = plt.subplots(figsize=(18, 9))
    _apply_dark_style(fig, [ax])

    display = retention_pct.copy()
    display.index = display.index.astype(str)
    display.columns = display.columns.astype(str)

    sns.heatmap(
        display, annot=True, fmt=".0%", cmap=cmap,
        linewidths=0.3, linecolor=BG_DARK,
        cbar_kws={"label": "Retention Rate", "shrink": 0.6},
        ax=ax, annot_kws={"size": 7},
    )
    ax.set_xlabel("Cohort Index (months since first purchase)", fontsize=10, labelpad=8)
    ax.set_ylabel("Cohort Month", fontsize=10, labelpad=8)
    ax.tick_params(colors=SUB_CLR, labelsize=8)

    _title(fig, "Cohort Retention Analysis","% of customers retained by month since acquisition")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────
# 13. PRODUCT PAIR FREQUENCY (EXTRA)
# ─────────────────────────────────────────────

def plot_product_pairs(df: pd.DataFrame, top_n: int = 15) -> plt.Figure:
    invoice_product = (
        df.groupby(["InvoiceNo", "Description"])["Quantity"]
        .sum().unstack(fill_value=0)
    )
    invoice_product = (invoice_product > 0).astype(int)

    pair_counts = Counter()
    for _, row in invoice_product.iterrows():
        items = row[row == 1].index.tolist()
        if len(items) >= 2:
            pair_counts.update(combinations(items, 2))

    if not pair_counts:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, "Not enough co-purchase data",
                ha="center", va="center", color=TEXT_CLR)
        return fig

    pairs_df = (
        pd.DataFrame(pair_counts.items(), columns=["Pair", "Count"])
        .sort_values("Count", ascending=False)
        .head(top_n)
    )
    pairs_df["Label"] = pairs_df["Pair"].apply(
        lambda p: f"{str(p[0])[:22]}… + {str(p[1])[:22]}…"
        if (len(str(p[0])) > 22 or len(str(p[1])) > 22)
        else f"{p[0]} + {p[1]}"
    )

    fig, ax = plt.subplots(figsize=(16, 7))
    _apply_dark_style(fig, [ax])
    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(pairs_df))]
    _bar(ax, pairs_df["Label"], pairs_df["Count"], colors=bar_colors)
    ax.set_xticklabels(pairs_df["Label"], rotation=50, ha="right", fontsize=7.5)
    ax.set_ylabel("Co-purchase Count", fontsize=10)
    ax.set_title(f"Top {top_n} Most Frequently Bought Together", color=TEXT_CLR, fontsize=13)

    _title(fig, "Product Pair Analysis", "Items most often purchased in the same order")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return fig


# ─────────────────────────────────────────────
# 14. EXECUTIVE SUMMARY DASHBOARD (EXTRA)
# ─────────────────────────────────────────────

def plot_executive_dashboard(df: pd.DataFrame, rfm: pd.DataFrame,segment_summary: pd.DataFrame) -> plt.Figure:
    df = df.copy()
    df["MonthName"] = df["InvoiceDate"].dt.month.map(
        lambda m: calendar.month_abbr[int(m)]
    )
    month_order = list(calendar.month_abbr)[1:]
    monthly_rev = df.groupby("MonthName")["TotalPrice"].sum().reindex(
        [m for m in month_order if m in df["MonthName"].unique()]
    )

    product_sales = df.groupby("Description")["TotalPrice"].sum().nlargest(10)
    country_rev   = df.groupby("Country")["TotalPrice"].sum().nlargest(10)
    seg_rev       = rfm.groupby("Segment")["Monetary"].sum().sort_values(ascending=False)

    fig = plt.figure(figsize=(20, 13))
    _apply_dark_style(fig)
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.4)

    # Monthly trend
    ax1 = fig.add_subplot(gs[0, :])
    _apply_dark_style(fig, [ax1])
    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(monthly_rev))]
    ax1.bar(monthly_rev.index, monthly_rev.values, color=bar_colors, alpha=0.8, edgecolor="none")
    ax1.plot(monthly_rev.index, monthly_rev.values, color=ACCENT2, linewidth=3,marker="o", markersize=8, markerfacecolor=BG_DARK,markeredgecolor=ACCENT2, markeredgewidth=2, zorder=5)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
    ax1.set_title("Monthly Revenue Trend", color=TEXT_CLR, fontsize=13)

    # Segment count
    ax2 = fig.add_subplot(gs[1, 0])
    _apply_dark_style(fig, [ax2])
    sc = rfm["Segment"].value_counts()
    _bar(ax2, sc.index, sc.values, colors=[PALETTE[i % len(PALETTE)] for i in range(len(sc))])
    ax2.set_xticklabels(sc.index, rotation=30, ha="right", fontsize=7)
    ax2.set_title("Customer Segments", color=TEXT_CLR, fontsize=11)

    # Top products
    ax3 = fig.add_subplot(gs[1, 1:])
    _apply_dark_style(fig, [ax3])
    labels = [d[:25] + "…" if len(d) > 25 else d for d in product_sales.index]
    _bar(ax3, labels, product_sales.values,colors=[PALETTE[i % len(PALETTE)] for i in range(10)])
    ax3.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax3.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"£{x/1e3:.0f}K"))
    ax3.set_title("Top 10 Products by Revenue", color=TEXT_CLR, fontsize=11)

    # Country donut
    ax4 = fig.add_subplot(gs[2, 0])
    top5c  = country_rev.head(5)
    other_c= country_rev.iloc[5:].sum()
    _donut(ax4, top5c.values.tolist() + [other_c],top5c.index.tolist() + ["Others"],center_text="Markets")
    ax4.set_title("Revenue by Country", color=TEXT_CLR, fontsize=11, pad=8)

    # Segment revenue donut
    ax5 = fig.add_subplot(gs[2, 1])
    _donut(ax5, seg_rev.values, seg_rev.index,colors=[PALETTE[i % len(PALETTE)] for i in range(len(seg_rev))],center_text="Segments")
    ax5.set_title("Revenue by Segment", color=TEXT_CLR, fontsize=11, pad=8)

    # Scatter recency × monetary
    ax6 = fig.add_subplot(gs[2, 2])
    _apply_dark_style(fig, [ax6])
    for i, seg in enumerate(rfm["Segment"].unique()):
        mask = rfm["Segment"] == seg
        ax6.scatter(rfm.loc[mask, "Recency"], rfm.loc[mask, "Monetary"],
                    color=PALETTE[i % len(PALETTE)], alpha=0.45, s=15, label=seg)
    ax6.set_xlabel("Recency (days)", fontsize=8)
    ax6.set_ylabel("Monetary (£)", fontsize=8)
    ax6.set_title("Recency vs Monetary", color=TEXT_CLR, fontsize=11)
    ax6.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x/1e3:.0f}K"))
    ax6.legend(fontsize=6, facecolor=BG_CARD, edgecolor=GRID_CLR,labelcolor=TEXT_CLR, markerscale=1.5)

    _title(fig, "Executive Dashboard","Monthly revenue · Segments · Products · Countries · Scatter")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return fig


# ─────────────────────────────────────────────
# CONVENIENCE: render all plots
# ─────────────────────────────────────────────

def render_all(results: dict, show: bool = True, save_dir: str = None):
    """
    Call every plot function and optionally save PNGs.

    Args:
        results  : dict returned by data_processing.run_pipeline()
        show     : call plt.show() after each figure
        save_dir : if set, saves each figure as <save_dir>/<name>.png
    """
    import os

    df              = results["df"]
    rfm             = results["rfm"]
    segment_summary = results["segment_summary"]
    retention_pct   = results["retention_pct"]

    plots = {
        "01_monthly_sales":       plot_monthly_sales(df),
        "02_country_analysis":    plot_country_analysis(df),
        "03_top_products":        plot_top_products(df),
        "04_rfm_overview":        plot_rfm_overview(rfm),
        "05_rfm_heatmap":         plot_rfm_heatmap(rfm),
        "06_segment_behaviour":   plot_segment_behaviour(df, rfm),
        "07_order_value":         plot_order_value_distribution(df),
        "08_basket_size":         plot_basket_size(df),
        "09_time_patterns":       plot_time_patterns(df),
        "10_retention_churn":     plot_retention_churn(df, rfm),
        "11_kpi_dashboard":       plot_kpi_dashboard(df, rfm),
        "12_cohort_retention":    plot_cohort_retention(retention_pct),
        "13_product_pairs":       plot_product_pairs(df),
        "14_exec_dashboard":      plot_executive_dashboard(df, rfm, segment_summary),
    }

    for name, fig in plots.items():
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{name}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            print(f"Saved → {path}")
        if show:
            plt.show()
        plt.close(fig)

    return plots
