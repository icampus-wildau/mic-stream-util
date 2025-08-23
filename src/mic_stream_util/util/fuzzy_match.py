"""Fuzzy matching utilities for device discovery."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from fuzzywuzzy import fuzz


def find_best_match(query: str, candidates: List[Dict[str, Any]], threshold: int = 70) -> Optional[Tuple[int, Dict[str, Any]]]:
    """
    Find the best matching device using fuzzy string matching.

    Args:
        query: The search query (device name)
        candidates: List of device dictionaries
        threshold: Minimum similarity score (0-100)

    Returns:
        Tuple of (index, device_dict) or None if no match found
    """

    if not candidates:
        return None

    best_score = 0
    best_match = None
    best_index = -1

    for i, device in enumerate(candidates):
        # Try matching against device name
        name = device.get("name", "")
        score = fuzz.ratio(query.lower(), name.lower())

        # Also try partial matching
        partial_score = fuzz.partial_ratio(query.lower(), name.lower())
        score = max(score, partial_score)

        if score > best_score:
            best_score = score
            best_match = device
            best_index = i

    if best_score >= threshold and best_match is not None:
        return best_index, best_match

    return None


class FuzzySearch:
    @staticmethod
    def find_best_match(query: str, candidates: list[str], threshold: int = 70) -> tuple[int, str] | tuple[None, None]:
        """
        Find the best matching string using fuzzy string matching.

        Args:
            query: The search query (device name)
            candidates: List of strings
            threshold: Minimum similarity score (0-100)

        Returns:
            Tuple of (index, name) or None if no match found
        """
        if not candidates:
            return None

        best_score = 0
        best_match = None
        best_index = -1

        for i, candidate in enumerate(candidates):
            score = fuzz.ratio(query.lower(), candidate.lower())
            partial_score = fuzz.partial_ratio(query.lower(), candidate.lower())
            score = max(score, partial_score)

            if score > best_score:
                best_score = score
                best_match = candidate
                best_index = i

        if best_score >= threshold and best_match is not None:
            return best_index, best_match

        return None, None
