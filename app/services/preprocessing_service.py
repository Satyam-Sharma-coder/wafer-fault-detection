import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import KNNImputer

class PreprocessingService:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.imputer = KNNImputer(n_neighbors=5)

    def preprocess(self, df: pd.DataFrame):
        # 1. Drop Time column
        if "Time" in df.columns:
            df = df.drop(columns=["Time"])

        # 2. Separate label
        y = df["Pass/Fail"]
        X = df.drop(columns=["Pass/Fail"])

        # 3. Drop high-null columns (>50%)
        X = X.dropna(axis=1, thresh=int(0.5 * len(X)))

        # 4. Drop constant columns
        X = X.loc[:, X.nunique() > 1]

        # 5. Impute missing values
        X_imputed = self.imputer.fit_transform(X)

        # 6. Normalize
        X_scaled = self.scaler.fit_transform(X_imputed)

        return X_scaled, y.to_numpy()
