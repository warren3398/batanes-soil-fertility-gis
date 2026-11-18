"""BSWM Map Guidebook Tables 17–20 and optional user-defined classes."""

import math

import numpy as np

STANDARD_CLASSES = [
    ("Low", "#ff0000"), ("Moderately Low", "#ffaa00"),
    ("Moderately High", "#e6e600"), ("High", "#4ce600"),
]
CUSTOM_CLASSES = [
    ("Very low", "#e74c43"), ("Low", "#f29435"), ("Moderate", "#f4d64f"),
    ("High", "#91ce53"), ("Very high", "#22864b"),
]
SOURCE = "BSWM Map Guidebook, Tables 17–20"
RULES = {
    "N": ("%", "Table 17", [0, .1025, .1775, .2275], None,
          ["0–0.10", "0.105–0.175", "0.18–0.225", "≥ 0.23"]),
    "OM": ("%", "Table 17", [0, 2.05, 3.55, 4.55], 5.5,
           ["0–2", "2.1–3.5", "3.6–4.5", "4.6–5.5"]),
    "Bray 1": ("ppm", "Table 18 · Bray 1", [0, 2.05, 6.05, 10.05], 15,
               ["0–2", "2.10–6.0", "6.1–10", "10.1–15"]),
    "Olsen": ("ppm", "Table 18 · Olsen", [0, 6.05, 10.05, 15.05], 20,
              ["0–6", "6.1–10", "10.1–15", "15.1–20"]),
    "Cold H2SO4": ("ppm", "Table 19 · Cold H₂SO₄", [0, 35.5, 55.5, math.nextafter(75., math.inf)], None,
                   ["0–35", "36–55", "56–75", "> 75"]),
    "Hot H2SO4": ("ppm", "Table 19 · Hot H₂SO₄", [0, 75.5, 113.5, math.nextafter(150., math.inf)], None,
                  ["0–75", "76–113", "114–150", "> 150"]),
    "pH": ("pH (1:1)", "Table 20 · 1:1 ratio", [0, 4.55, 5.05, 5.55, 6.85], 14,
           ["≤ 4.5 or ≥ 6.9", "4.6–5.0", "5.1–5.5", "5.6–6.8"]),
}


def classify_values(values, rules):
    positions = np.searchsorted(rules["breaks"], values, side="right")
    codes = np.asarray(rules.get("class_codes", [1, 2, 3, 4, 5]), dtype="uint8")
    classes = codes[positions]
    ceiling = rules.get("class_max")
    if ceiling is not None:
        classes[values > ceiling] = 0
    return classes


def classification(summary, thresholds, p_method, k_method):
    if thresholds.strip():
        breaks = [float(value) for value in thresholds.split(",")]
        if len(breaks) != 4 or not np.isfinite(breaks).all() or not (np.diff(breaks) > 0).all():
            raise ValueError("Enter four increasing numeric class thresholds")
        ranges = [f"< {breaks[0]:.6g}",
                  *[f"{a:.6g} to < {b:.6g}" for a, b in zip(breaks, breaks[1:])],
                  f"≥ {breaks[-1]:.6g}"]
        summary.update(breaks=breaks, classification="Custom thresholds")
        return zip(CUSTOM_CLASSES, ranges)

    parameter = summary["parameter"]
    method = p_method if parameter == "P" else k_method if parameter == "K" else parameter
    units, reference, breaks, maximum, ranges = RULES[method]
    codes = [0, 1, 2, 3, 4, 1] if parameter == "pH" else [0, 1, 2, 3, 4]
    summary.update(units=units, breaks=breaks, class_codes=codes, class_max=maximum,
                   classification=f"BSWM Map Guidebook · {reference}", classification_source=SOURCE,
                   classification_note="Published ranges shown. Predictions between adjacent printed bands are split at their midpoint. Only classified cells are included in area shares.")
    return zip(STANDARD_CLASSES, ranges)
