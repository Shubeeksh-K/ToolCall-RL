"""Simple unit conversion tool."""

from __future__ import annotations


_CONVERSIONS = {
    ("meters", "feet"): lambda value: value * 3.28084,
    ("feet", "meters"): lambda value: value / 3.28084,
    ("kilometers", "miles"): lambda value: value * 0.621371,
    ("miles", "kilometers"): lambda value: value / 0.621371,
    ("kilograms", "pounds"): lambda value: value * 2.20462,
    ("pounds", "kilograms"): lambda value: value / 2.20462,
    ("celsius", "fahrenheit"): lambda value: (value * 9 / 5) + 32,
    ("fahrenheit", "celsius"): lambda value: (value - 32) * 5 / 9,
}


def unit_converter(value: float, from_unit: str, to_unit: str) -> dict[str, str | float]:
    """Convert a number between a small set of common units."""

    normalized_from_unit = from_unit.strip().lower()
    normalized_to_unit = to_unit.strip().lower()

    if normalized_from_unit == normalized_to_unit:
        return {
            "value": value,
            "from_unit": normalized_from_unit,
            "to_unit": normalized_to_unit,
            "result": value,
        }

    conversion = _CONVERSIONS.get((normalized_from_unit, normalized_to_unit))
    if conversion is None:
        return {
            "value": value,
            "from_unit": normalized_from_unit,
            "to_unit": normalized_to_unit,
            "error": "Unsupported conversion.",
        }

    return {
        "value": value,
        "from_unit": normalized_from_unit,
        "to_unit": normalized_to_unit,
        "result": round(conversion(value), 4),
    }
