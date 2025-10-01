"""
ShadCN-inspired component library for Streamlit GoSales Engine
Provides reusable UI components with consistent styling
"""
from __future__ import annotations
from typing import Optional, Literal
import streamlit as st
import pandas as pd


# Color palette (Grayscale + #BAD532)
class Colors:
    # Primary colors - #BAD532
    PRIMARY = "#BAD532"
    PRIMARY_DARK = "#9BB828"
    PRIMARY_LIGHT = "#C9E05C"
    
    # Secondary colors - Grayscale
    SECONDARY = "#D9D9D9"
    SECONDARY_DARK = "#BFBFBF"
    SECONDARY_LIGHT = "#EDEDED"
    
    # Semantic colors
    SUCCESS = "#BAD532"  # Use brand color for success
    WARNING = "#E6C229"
    ERROR = "#DC2626"
    INFO = "#666666"
    
    # Neutral colors - Grayscale
    BACKGROUND = "#FFFFFF"
    BACKGROUND_DARK = "#141414"
    SURFACE = "#F7F7F7"
    SURFACE_DARK = "#1F1F1F"
    BORDER = "#E0E0E0"
    TEXT = "#1A1A1A"
    TEXT_MUTED = "#737373"


def card(
    title: Optional[str] = None,
    content: Optional[str] = None,
    icon: Optional[str] = None,
    variant: Literal["default", "bordered", "elevated"] = "default",
    color: Optional[str] = None
):
    """
    Render a ShadCN-style card component
    
    Args:
        title: Card title
        content: Card content (markdown supported)
        icon: Optional emoji icon
        variant: Card style variant
        color: Optional border color
    """
    border_style = ""
    shadow = ""
    
    if variant == "bordered":
        border_color = color or Colors.BORDER
        border_style = f"border: 2px solid {border_color};"
    elif variant == "elevated":
        shadow = "box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);"
    
    title_html = ""
    if title:
        icon_html = f'<span style="margin-right: 8px;">{icon}</span>' if icon else ""
        title_html = f"""
        <div style="
            font-size: 1.125rem;
            font-weight: 600;
            color: {Colors.TEXT};
            margin-bottom: 12px;
            display: flex;
            align-items: center;
        ">
            {icon_html}{title}
        </div>
        """
    
    content_html = ""
    if content:
        content_html = f"""
        <div style="
            color: {Colors.TEXT_MUTED};
            font-size: 0.875rem;
            line-height: 1.5;
        ">
            {content}
        </div>
        """
    
    card_html = f"""
    <div style="
        background-color: {Colors.BACKGROUND};
        border-radius: 8px;
        padding: 20px;
        margin: 12px 0;
        {border_style}
        {shadow}
        transition: all 0.2s ease-in-out;
    ">
        {title_html}
        {content_html}
    </div>
    """
    
    st.markdown(card_html, unsafe_allow_html=True)


def metric_card(
    label: str,
    value: str,
    delta: Optional[str] = None,
    delta_color: Literal["normal", "inverse", "off"] = "normal",
    icon: Optional[str] = None,
    help_text: Optional[str] = None
):
    """
    Enhanced metric card with ShadCN styling
    
    Args:
        label: Metric label
        value: Metric value
        delta: Change indicator (e.g., "+12%")
        delta_color: Delta color scheme
        icon: Optional emoji icon
        help_text: Optional tooltip text
    """
    # Determine delta color
    delta_html = ""
    if delta:
        # Try to determine if delta is positive/negative
        try:
            # Remove common symbols and try to convert to float
            cleaned_delta = delta.replace("%", "").replace("$", "").replace(",", "").strip()
            is_positive = delta.startswith("+") or (not delta.startswith("-") and float(cleaned_delta) > 0)
        except (ValueError, AttributeError):
            # If conversion fails, it's probably descriptive text - default to neutral
            is_positive = None
        
        if delta_color == "normal":
            if is_positive is None:
                color = Colors.TEXT_MUTED
            else:
                color = Colors.SUCCESS if is_positive else Colors.ERROR
        elif delta_color == "inverse":
            if is_positive is None:
                color = Colors.TEXT_MUTED
            else:
                color = Colors.ERROR if is_positive else Colors.SUCCESS
        else:
            color = Colors.TEXT_MUTED
        
        delta_html = f'<div style="color: {color}; font-size: 0.875rem; font-weight: 500; margin-top: 4px;">{delta}</div>'
    
    icon_html = f'<div style="font-size: 1.5rem; margin-bottom: 8px;">{icon}</div>' if icon else ""
    
    help_html = f'<div style="color: {Colors.TEXT_MUTED}; font-size: 0.7rem; margin-top: 4px;" title="{help_text}">ℹ️</div>' if help_text else ""
    
    # Compact, modern metric card with grayscale + #BAD532 accent
    metric_html = f'''<div style="background: {Colors.SURFACE}; border: 1px solid {Colors.BORDER}; border-radius: 6px; padding: 12px 16px; margin: 8px 0; transition: all 0.2s;">
        {icon_html}
        <div style="color: {Colors.TEXT_MUTED}; font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px;">{label}</div>
        <div style="color: {Colors.TEXT}; font-size: 1.5rem; font-weight: 600; line-height: 1.2;">{value}</div>
        {delta_html}
        {help_html}
    </div>'''
    
    st.markdown(metric_html, unsafe_allow_html=True)


def alert(
    message: str,
    variant: Literal["info", "success", "warning", "error"] = "info",
    title: Optional[str] = None,
    dismissible: bool = False
):
    """
    Render an alert/notification component
    
    Args:
        message: Alert message
        variant: Alert type
        title: Optional alert title
        dismissible: Whether alert can be dismissed
    """
    colors = {
        "info": (Colors.INFO, "#dbeafe", "#1e40af"),
        "success": (Colors.SUCCESS, "#d1fae5", "#065f46"),
        "warning": (Colors.WARNING, "#fef3c7", "#92400e"),
        "error": (Colors.ERROR, "#fee2e2", "#991b1b")
    }
    
    border_color, bg_color, text_color = colors[variant]
    
    icons = {
        "info": "ℹ️",
        "success": "✅",
        "warning": "⚠️",
        "error": "❌"
    }
    
    icon = icons[variant]
    
    title_html = ""
    if title:
        title_html = f"""
        <div style="
            font-weight: 600;
            font-size: 0.95rem;
            margin-bottom: 4px;
            color: {text_color};
        ">
            {title}
        </div>
        """
    
    alert_html = f"""
    <div style="
        background-color: {bg_color};
        border-left: 4px solid {border_color};
        border-radius: 6px;
        padding: 16px;
        margin: 12px 0;
        display: flex;
        align-items: start;
        gap: 12px;
    ">
        <div style="font-size: 1.25rem; line-height: 1; margin-top: 2px;">
            {icon}
        </div>
        <div style="flex: 1;">
            {title_html}
            <div style="
                color: {text_color};
                font-size: 0.875rem;
                line-height: 1.5;
            ">
                {message}
            </div>
        </div>
    </div>
    """
    
    st.markdown(alert_html, unsafe_allow_html=True)


def badge(
    text: str,
    variant: Literal["default", "primary", "success", "warning", "error", "info"] = "default"
):
    """
    Render a badge component
    
    Args:
        text: Badge text
        variant: Badge color variant
    """
    colors = {
        "default": (Colors.SECONDARY, Colors.BACKGROUND),
        "primary": (Colors.PRIMARY, Colors.BACKGROUND),
        "success": (Colors.SUCCESS, Colors.BACKGROUND),
        "warning": (Colors.WARNING, Colors.TEXT),
        "error": (Colors.ERROR, Colors.BACKGROUND),
        "info": (Colors.INFO, Colors.BACKGROUND)
    }
    
    bg_color, text_color = colors[variant]
    
    badge_html = f"""
    <span style="
        display: inline-flex;
        align-items: center;
        background-color: {bg_color};
        color: {text_color};
        font-size: 0.75rem;
        font-weight: 600;
        padding: 4px 10px;
        border-radius: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    ">
        {text}
    </span>
    """
    
    st.markdown(badge_html, unsafe_allow_html=True)


def stat_grid(stats: list[dict]):
    """
    Render a grid of statistics
    
    Args:
        stats: List of stat dicts with keys: label, value, delta, icon
    """
    cols = st.columns(len(stats))
    
    for col, stat in zip(cols, stats):
        with col:
            metric_card(
                label=stat.get("label", ""),
                value=stat.get("value", ""),
                delta=stat.get("delta"),
                icon=stat.get("icon")
            )


def data_table_enhanced(
    df: pd.DataFrame,
    title: Optional[str] = None,
    searchable: bool = True,
    downloadable: bool = True,
    page_size: int = 10
):
    """
    Enhanced data table with search, pagination, and export
    
    Args:
        df: DataFrame to display
        title: Optional table title
        searchable: Enable search functionality
        downloadable: Enable download button
        page_size: Rows per page
    """
    if title:
        st.markdown(f"### {title}")
    
    # Search functionality
    if searchable and not df.empty:
        search = st.text_input("🔍 Search table", key=f"search_{id(df)}")
        if search:
            # Search across all columns
            mask = df.astype(str).apply(lambda x: x.str.contains(search, case=False, na=False)).any(axis=1)
            df = df[mask]
    
    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        height=min(400, (len(df) + 1) * 35)
    )
    
    # Download button
    if downloadable and not df.empty:
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"{title.lower().replace(' ', '_') if title else 'data'}.csv",
            mime="text/csv"
        )


def progress_bar(value: float, max_value: float = 100, label: Optional[str] = None):
    """
    Enhanced progress bar with label
    
    Args:
        value: Current value
        max_value: Maximum value
        label: Optional label
    """
    percentage = (value / max_value) * 100
    
    label_html = ""
    if label:
        label_html = f"""
        <div style="
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            font-size: 0.875rem;
            color: {Colors.TEXT_MUTED};
        ">
            <span>{label}</span>
            <span>{percentage:.1f}%</span>
        </div>
        """
    
    progress_html = f"""
    <div>
        {label_html}
        <div style="
            width: 100%;
            height: 8px;
            background-color: {Colors.SURFACE};
            border-radius: 4px;
            overflow: hidden;
        ">
            <div style="
                width: {percentage}%;
                height: 100%;
                background: linear-gradient(90deg, {Colors.PRIMARY} 0%, {Colors.PRIMARY_LIGHT} 100%);
                transition: width 0.3s ease-in-out;
            "></div>
        </div>
    </div>
    """
    
    st.markdown(progress_html, unsafe_allow_html=True)


def tabs_navigation(tabs: list[str], active_tab: Optional[str] = None) -> str:
    """
    Modern tab navigation component
    
    Args:
        tabs: List of tab names
        active_tab: Currently active tab
        
    Returns:
        Selected tab name
    """
    if active_tab is None:
        active_tab = tabs[0]
    
    # Use Streamlit's built-in tabs with custom styling
    return st.radio(
        "Navigation",
        tabs,
        index=tabs.index(active_tab) if active_tab in tabs else 0,
        horizontal=True,
        label_visibility="collapsed"
    )


def skeleton_loader(lines: int = 3):
    """
    Display a loading skeleton
    
    Args:
        lines: Number of skeleton lines to display
    """
    skeleton_html = ""
    for i in range(lines):
        width = 100 if i < lines - 1 else 60  # Last line is shorter
        skeleton_html += f"""
        <div style="
            height: 16px;
            background: linear-gradient(90deg, {Colors.SURFACE} 0%, {Colors.BORDER} 50%, {Colors.SURFACE} 100%);
            background-size: 200% 100%;
            animation: shimmer 1.5s infinite;
            border-radius: 4px;
            margin: 8px 0;
            width: {width}%;
        "></div>
        """
    
    skeleton_container = f"""
    <style>
        @keyframes shimmer {{
            0% {{ background-position: 200% 0; }}
            100% {{ background-position: -200% 0; }}
        }}
    </style>
    <div style="padding: 20px;">
        {skeleton_html}
    </div>
    """
    
    st.markdown(skeleton_container, unsafe_allow_html=True)

