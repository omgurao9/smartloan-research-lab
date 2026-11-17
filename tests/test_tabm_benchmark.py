import numpy as np
import pandas as pd

from research_lab.baseline import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from research_lab.tabm_benchmark import _encode_features


def test_tabm_preprocessing_fits_maps_on_training_and_handles_unknown_categories():
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    train = pd.DataFrame([
        {**{c: float(i + 1) for i, c in enumerate(NUMERIC_FEATURES)},
         **{c: "A" if i % 2 else "B" for i, c in enumerate(CATEGORICAL_FEATURES)}}
        for _ in range(4)
    ])
    other = train.iloc[:2].copy()
    other.loc[other.index[0], "person_gender"] = "UNSEEN"
    encoded = _encode_features(train, other, features)
    x_num_train, x_num_other, cardinalities, metadata, x_cat_train, x_cat_other = encoded

    assert x_num_train.shape == (4, len(NUMERIC_FEATURES))
    assert x_num_other.shape == (2, len(NUMERIC_FEATURES))
    assert x_cat_train.shape == (4, len(CATEGORICAL_FEATURES))
    assert x_cat_other.shape == (2, len(CATEGORICAL_FEATURES))
    assert len(cardinalities) == len(CATEGORICAL_FEATURES)
    assert x_cat_other[0, 0] == 0
    assert metadata["numeric_features"] == NUMERIC_FEATURES
    assert np.isfinite(x_num_train).all()
    assert np.isfinite(x_num_other).all()
