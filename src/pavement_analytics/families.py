"""Editable analytical groupings of distress codes; not causal classifications."""

DEFAULT_FAMILIES = {
    "pcc_freeze_thaw": ["64", "70", "74", "75"],
    "pcc_thermal": ["61", "63", "65"],
    "pcc_moisture": ["69", "71", "76"],
    "pcc_load": ["62", "72"],
    "pcc_other": ["66", "67", "68", "73"],
    "ac_thermal_crack": ["43", "47", "48"],
    "ac_aging": ["52", "57"],
    "ac_high_temp": ["42", "53", "54"],
    "ac_load": ["41", "45"],
    "ac_subgrade": ["44", "56"],
    "ac_other": ["46", "49", "50", "51", "55"],
}


def default_families(codes) -> dict[str, list[str]]:
    """Keep all named slots and preserve unknown codes as individual families."""
    from .security import SafeError
    import re
    codes = {str(c) for c in codes}
    if any(not re.fullmatch(r"[0-9]{2}", c) for c in codes):
        raise SafeError("INVALID_DEDUCT_CODE")
    result = {name: list(values) for name, values in DEFAULT_FAMILIES.items()}
    covered = {c for values in result.values() for c in values}
    result.update({"deduct_" + c: [c] for c in sorted(codes - covered)})
    return result
