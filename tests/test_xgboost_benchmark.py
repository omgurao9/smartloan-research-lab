import numpy as np
import pytest

from research_lab.split import prepare_split
from research_lab.xgboost_benchmark import run_xgboost_comparison
from test_boosting import make_frame


def test_xgboost_benchmark_keeps_test_partition_unopened():
    pytest.importorskip("xgboost")
    frame = make_frame()
    splits, _ = prepare_split(
        frame, train_fraction=0.6, validation_fraction=0.2, test_fraction=0.2
    )
    results, metadata, models = run_xgboost_comparison(frame, splits, seed=7)
    assert len(results) == 2
    assert set(results["model"]) == {"xgboost"}
    assert metadata["test_partition_used"] is False
    assert len(models) == 2
    assert np.isfinite(results["average_precision"]).all()
