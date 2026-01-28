"""
Subgen GUI Styling Module
Modern macOS Big Sur inspired design system
"""

# Color Palette
# Minimalist Monochrome Terminal
COLORS = {
    'background': '#0f0f0f',
    'card': '#0f0f0f',            # Blend with background
    'text': '#eeeeee',            # Off-white for less eye strain
    'text_secondary': '#666666',  # Darker grey for secondary
    'primary': '#ffffff',         # White for active elements
    'primary_hover': '#cccccc',
    'border': '#333333',          # Subtle dark grey border
    'accent': '#000000',
    'progress': '#33ff00',
}

# Monospace font
FONT_FAMILY = "Menlo, Monaco, 'Courier New', monospace"

# Global Application Style
APP_STYLE = f"""
QMainWindow {{
    background-color: {COLORS['background']};
}}

QWidget {{
    font-family: {FONT_FAMILY};
    font-size: 12px;
    color: {COLORS['text']};
}}

QSplitter::handle {{
    background-color: {COLORS['border']};
    width: 1px;
}}
"""

CARD_STYLE = f"""
    QFrame {{
        background-color: {COLORS['background']};
        border: none;
    }}
"""

PRIMARY_BUTTON_STYLE = f"""
    QPushButton {{
        background-color: {COLORS['text']};
        color: {COLORS['background']};
        border: 1px solid {COLORS['text']};
        padding: 8px 16px;
        font-weight: bold;
        font-family: {FONT_FAMILY};
        border-radius: 0px;
    }}
    QPushButton:hover {{
        background-color: {COLORS['primary_hover']};
        border-color: {COLORS['primary_hover']};
    }}
    QPushButton:pressed {{
        background-color: {COLORS['text_secondary']};
    }}
    QPushButton:disabled {{
        background-color: {COLORS['border']};
        color: {COLORS['text_secondary']};
        border-color: {COLORS['border']};
    }}
"""

SECONDARY_BUTTON_STYLE = f"""
    QPushButton {{
        background-color: transparent;
        color: {COLORS['text']};
        border: 1px solid {COLORS['text_secondary']};
        padding: 6px 12px;
        font-family: {FONT_FAMILY};
        border-radius: 0px;
    }}
    QPushButton:hover {{
        border-color: {COLORS['text']};
    }}
    QPushButton:pressed {{
        background-color: {COLORS['border']};
    }}
"""

DROP_ZONE_STYLE = f"""
    QListWidget {{
        border: 1px solid {COLORS['border']};
        background-color: {COLORS['background']};
        border-radius: 0px;
        color: {COLORS['text']};
        font-family: {FONT_FAMILY};
    }}
    QListWidget::item {{
        padding: 8px;
        border-bottom: 1px solid {COLORS['border']};
    }}
    QListWidget::item:selected {{
        background-color: {COLORS['border']};
        color: {COLORS['text']};
    }}
"""

RADIO_BUTTON_STYLE = f"""
    QRadioButton {{
        spacing: 8px;
        color: {COLORS['text']};
        font-family: {FONT_FAMILY};
    }}
    QRadioButton::indicator {{
        width: 12px;
        height: 12px;
        border: 1px solid {COLORS['text_secondary']};
        border-radius: 6px; /* Circle for minimalist look */
        background: transparent;
    }}
    QRadioButton::indicator:checked {{
        background-color: {COLORS['text']};
        border: 1px solid {COLORS['text']};
    }}
"""

# Minimalist Tab
TAB_STYLE = f"""
    QTabWidget::pane {{
        border: 1px solid {COLORS['border']};
        background: {COLORS['background']};
        border-top: 1px solid {COLORS['border']};
    }}
    QTabBar::tab {{
        background: {COLORS['background']};
        border: none;
        padding: 8px 16px;
        color: {COLORS['text_secondary']};
        font-family: {FONT_FAMILY};
        text-transform: uppercase;
    }}
    QTabBar::tab:selected {{
        color: {COLORS['text']};
        font-weight: bold;
        border-bottom: 2px solid {COLORS['text']};
    }}
    QTabBar::tab:hover {{
        color: {COLORS['text']};
    }}
"""

PROGRESS_BAR_STYLE = f"""
    QProgressBar {{
        border: 1px solid {COLORS['border']};
        border-radius: 0px;
        text-align: center;
        background-color: {COLORS['background']};
        color: transparent;
    }}
    QProgressBar::chunk {{
        background-color: {COLORS['progress']};
    }}
"""

TREE_VIEW_STYLE = f"""
    QTreeView {{
        background-color: {COLORS['background']};
        border: none;
        color: {COLORS['text']};
        font-family: {FONT_FAMILY};
    }}
    QTreeView::item {{
        padding: 4px;
        border: none;
    }}
    QTreeView::item:selected {{
        background-color: {COLORS['border']};
        color: {COLORS['text']};
    }}
    QHeaderView::section {{
        background-color: {COLORS['background']};
        color: {COLORS['text_secondary']};
        border: none;
        border-bottom: 1px solid {COLORS['border']};
        padding: 4px;
        font-family: {FONT_FAMILY};
        font-weight: normal;
    }}
"""

LOG_STYLE = f"""
    QTextEdit {{
        background-color: {COLORS['background']};
        color: {COLORS['text_secondary']};
        border: 1px solid {COLORS['border']};
        font-family: {FONT_FAMILY};
        font-size: 11px;
        border-top: none;
    }}
"""

COMBOBOX_STYLE = ""

def get_full_stylesheet():
    """Get the complete application stylesheet"""
    return (
        APP_STYLE +
        DROP_ZONE_STYLE +
        PRIMARY_BUTTON_STYLE +
        PROGRESS_BAR_STYLE +
        COMBOBOX_STYLE
    )
