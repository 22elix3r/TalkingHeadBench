"""
Shared grader utilities for TalkingHeadBench.

These helpers are used across multiple graders (Sub-env 1, 2, and 3).
Define once, import everywhere.
"""


def set_f1(predicted: set, true: set) -> float:
    """Compute F1 score over sets with correct empty-set handling.

    Both ``predicted`` and ``true`` should contain comparable elements
    (e.g. strings or integers).  The function handles the degenerate
    cases where one or both sets are empty before attempting division.

    Args:
        predicted: The set of items predicted/recommended by the agent.
        true: The ground-truth set of items from the expert oracle.

    Returns:
        A float in [0.0, 1.0]:
        - 1.0 if both sets are empty (both sides agree nothing to report).
        - 0.0 if exactly one set is empty (one side missed everything).
        - The harmonic mean of precision and recall otherwise.
    """
    if not predicted and not true:
        return 1.0   # both agree: nothing to report
    if not predicted or not true:
        return 0.0   # one side empty, the other isn't
    precision = len(predicted & true) / len(predicted)
    recall    = len(predicted & true) / len(true)
    return 2 * precision * recall / (precision + recall + 1e-8)


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute the Jaccard similarity index between two sets.

    The Jaccard index measures the proportion of elements shared by
    both sets relative to the total number of distinct elements across
    both sets.  Empty-set edge cases are handled symmetrically with
    ``set_f1``.

    Args:
        set_a: First set of elements.
        set_b: Second set of elements.

    Returns:
        A float in [0.0, 1.0]:
        - 1.0 if both sets are empty (vacuously identical).
        - 0.0 if exactly one set is empty (no overlap possible).
        - ``|set_a ∩ set_b| / |set_a ∪ set_b|`` otherwise.
    """
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)
