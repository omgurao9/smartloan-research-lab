import numpy as np

from research_lab.final_evaluation import _bootstrap_intervals, _metric_values


def test_final_metrics_include_confusion_matrix_and_specificity():
    y = np.array([0, 0, 1, 1])
    probabilities = np.array([0.1, 0.8, 0.7, 0.9])
    metrics = _metric_values(y, probabilities)
    assert metrics["true_negative"] == 1
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0
    assert metrics["true_positive"] == 2
    assert metrics["specificity_at_0_5"] == 0.5


def test_bootstrap_intervals_are_reproducible_and_ordered():
    y = np.array([0] * 20 + [1] * 10)
    probabilities = np.linspace(0.01, 0.99, len(y))
    first = _bootstrap_intervals(y, probabilities, seed=42, repeats=30)
    second = _bootstrap_intervals(y, probabilities, seed=42, repeats=30)
    assert first == second
    for interval in first.values():
        assert interval["lower_95"] <= interval["upper_95"]
