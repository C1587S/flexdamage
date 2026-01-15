"""
FlexDamage Diagnostics Module.

Provides professional visualization tools for damage function analysis.
"""

from .visualizer import DiagnosticVisualizer
from .maps import MapVisualizer
from .styles import (
    StyleConfig,
    get_style,
    apply_style,
    get_colormap,
    get_categorical_colors,
    PARAM_LABELS,
    GREEK_STATS,
    SCIENTIFIC_STYLE,
    PRESENTATION_STYLE,
)

__all__ = [
    # Main visualizers
    "DiagnosticVisualizer",
    "MapVisualizer",
    # Style system
    "StyleConfig",
    "get_style",
    "apply_style",
    "get_colormap",
    "get_categorical_colors",
    "PARAM_LABELS",
    "GREEK_STATS",
    "SCIENTIFIC_STYLE",
    "PRESENTATION_STYLE",
]
