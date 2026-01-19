import numpy as np
import pandas as pd
from app.services.preprocessing_service import PreprocessingService

def test_preprocessing_fit_transform():
    df = pd.DataFrame({
        "a": [1, 2, np.nan, 4],
        "b": [10, 20, 30, 40],
        "c": [np.nan, np.nan, np.nan, np.nan],  # will be dropped
    })
    pre = PreprocessingService()
    X, cols = pre.fit_transform(df)
    assert X.shape[0] == 4
    assert len(cols) == 2
    assert isinstance(X, np.ndarray)