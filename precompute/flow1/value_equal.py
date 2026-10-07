from __future__ import annotations


def ingredients_equal(left, right) -> bool:
    """Match items by ingredient_name, then every field equal. Order ignored."""
    if not isinstance(left, list) or not isinstance(right, list):
        return False
    if len(left) != len(right):
        return False
    remaining = list(right)
    for item in left:
        if not isinstance(item, dict):
            return False
        name = item.get("ingredient_name")
        found_i = None
        for index, other in enumerate(remaining):
            if isinstance(other, dict) and other.get("ingredient_name") == name:
                found_i = index
                break
        if found_i is None:
            return False
        other = remaining.pop(found_i)
        if item != other:
            return False
    return True


def _pair_frozenset(item):
    if isinstance(item, (list, tuple)) and len(item) == 2:
        left, right = item
        if isinstance(left, str) and isinstance(right, str):
            return frozenset((left, right))
    return None


def _normalize_pairs(value):
    if not isinstance(value, list):
        return None
    if len(value) == 2 and all(isinstance(item, str) for item in value):
        return [frozenset(value)]
    pairs = []
    for item in value:
        parsed = _pair_frozenset(item)
        if parsed is None:
            return None
        pairs.append(parsed)
    return pairs


def pairs_equal(left, right) -> bool:
    """Same pairs; order of pairs and names inside a pair ignored."""
    left_pairs = _normalize_pairs(left)
    right_pairs = _normalize_pairs(right)
    if left_pairs is None or right_pairs is None:
        return False
    return sorted(left_pairs, key=lambda item: tuple(sorted(item))) == sorted(
        right_pairs, key=lambda item: tuple(sorted(item))
    )


def values_equal(left, right) -> bool:
    """Dicts ignore key order; lists compared by content, not order."""
    if isinstance(left, dict) and isinstance(right, dict):
        if set(left) != set(right):
            return False
        return all(values_equal(left[key], right[key]) for key in left)
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return False
        remaining = list(right)
        for item in left:
            found = None
            for index, other in enumerate(remaining):
                if values_equal(item, other):
                    found = index
                    break
            if found is None:
                return False
            remaining.pop(found)
        return True
    return left == right
