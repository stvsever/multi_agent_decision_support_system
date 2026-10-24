"""
TOON (Token-Oriented Object Notation) Converter
===============================================

A lightweight utility to convert hierarchical JSON data into a token-efficient 
text representation for LLM consumption.

Key Features:
- Removes syntax noise (braces, quotes, commas)
- Uses indentation for hierarchy
- Compacts lists and leaf nodes
- Preserves information losslessly
"""

from typing import Any, List, Dict, Union

def json_to_toon(data: Any, indent_level: int = 0, indent_str: str = "  ") -> str:
    """
    Convert JSON-serializable data to TOON format.
    
    Args:
        data: Input data (dict, list, str, int, float, etc.)
        indent_level: Current indentation depth
        indent_str: String used for indentation (default: 2 spaces)
        
    Returns:
        String representation in TOON format
    """
    current_indent = indent_str * indent_level
    
    # 1. Handle Dictionaries
    if isinstance(data, dict):
        if not data:
            return "{}"
            
        lines = []
        for key, value in data.items():
            # Special case for leaf lists (e.g. key: [val1, val2])
            if isinstance(value, list) and not any(isinstance(x, (dict, list)) for x in value):
                # Compact list inline
                val_str = f"[{', '.join(str(v) for v in value)}]"
                lines.append(f"{current_indent}{key}: {val_str}")
                
            # Special case for COMPASS feature leaf objects
            # - raw UKB leaves: {'feature': ..., 'value': ..., 'z_score': ...}
            # - DataLoader flattened: {'field_name': ..., 'value': ..., 'z_score': ...}
            # Each leaf keeps its value, unit, deviation (when present) and
            # qualifiers: "Label = 3.12 cm3 (z -1.20) [ref range 3.0 to 4.2]".
            elif value and isinstance(value, list) and all(
                isinstance(x, dict) and ('feature' in x or 'field_name' in x)
                and ('z_score' in x or 'value' in x)
                for x in value
            ):
                lines.append(f"{current_indent}{key}:")
                item_strs = [_leaf_text(item) for item in value]
                
                # Split into multi-line if too long
                chunk_str = ", ".join(item_strs)
                if len(chunk_str) > 100:
                    for i_str in item_strs:
                        lines.append(f"{current_indent}{indent_str}- {i_str}")
                else:
                    lines.append(f"{current_indent}{indent_str}{chunk_str}")

            # Standard recursive handling
            elif isinstance(value, (dict, list)):
                lines.append(f"{current_indent}{key}:")
                lines.append(json_to_toon(value, indent_level + 1, indent_str))
            else:
                # Simple value
                lines.append(f"{current_indent}{key}: {value}")
                
        return "\n".join(lines)

    # 2. Handle Lists
    elif isinstance(data, list):
        if not data:
            return "[]"
            
        # Check if list of simple items
        if not any(isinstance(x, (dict, list)) for x in data):
            return f"{current_indent}[{', '.join(str(x) for x in data)}]"
            
        # List of complex objects
        lines = []
        for item in data:
            # If item is simple dict, try to keep it compact? 
            # For now, treat list items with a dash bullet
            item_str = json_to_toon(item, indent_level + 1, indent_str)
            # Remove the first level of indentation from the recursive call
            # because we add the dash at the current level
            stripped_item_str = item_str.lstrip()
            lines.append(f"{current_indent}- {stripped_item_str}")
            
        return "\n".join(lines)

    # 3. Handle Primitives
    else:
        return f"{current_indent}{str(data)}"


_LEAF_KEYS = {"feature", "field_name", "feature_id", "value", "unit", "z_score", "domain", "path_in_hierarchy", "qualifiers"}


def _leaf_text(item: Dict[str, Any]) -> str:
    """One feature leaf without loss: label, value and unit, deviation, qualifiers."""
    name = str(item.get("field_name") or item.get("feature") or item.get("feature_id") or "unknown")
    z = item.get("z_score")
    try:
        z_val = float(z) if z is not None else None
        if z_val is not None and z_val != z_val:
            z_val = None
    except (TypeError, ValueError):
        z_val = None
    parts = [name]
    raw = item.get("value")
    show_value = raw is not None and str(raw).strip() != ""
    if show_value and z_val is not None:
        try:
            show_value = abs(float(raw) - z_val) > 1e-9  # the z stood in for a missing value
        except (TypeError, ValueError):
            pass
    if show_value:
        unit = str(item.get("unit") or "").strip()
        parts.append(f"= {raw}{(' ' + unit) if unit else ''}")
    if z_val is not None:
        parts.append(f"(z {z_val:+.2f})")
    extras = {k: v for k, v in item.items() if k not in _LEAF_KEYS and not isinstance(v, (dict, list)) and v not in (None, "")}
    qualifiers = item.get("qualifiers")
    if isinstance(qualifiers, dict):
        extras.update({k: v for k, v in qualifiers.items() if v not in (None, "")})
    if extras:
        parts.append("[" + "; ".join(f"{str(k).replace('_', ' ')} {v}" for k, v in extras.items()) + "]")
    return " ".join(parts)
