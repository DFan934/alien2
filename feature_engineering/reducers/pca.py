############################
# feature_engineering/reducers/pca.py
############################
from __future__ import annotations

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class PCAReducer:
    """Leakage‑safe PCA: call fit on train DF, transform on val/test."""

    def __init__(self, explained_var: float = 0.95):
        self.explained_var = explained_var
        self.scaler = StandardScaler()
        self.pca = PCA(self.explained_var, svd_solver="full", random_state=42)

    # -----------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        scaled = self.scaler.fit_transform(df)
        comps = self.pca.fit_transform(scaled)
        meta = {
            "components": self.pca.components_,
            "explained_variance_ratio": self.pca.explained_variance_ratio_,
        }
        pc_df = pd.DataFrame(comps, index=df.index, columns=[f"PC{i+1}" for i in range(comps.shape[1])])
        return pc_df, meta

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        scaled = self.scaler.transform(df)
        comps = self.pca.transform(scaled)
        return pd.DataFrame(comps, index=df.index, columns=[f"PC{i+1}" for i in range(comps.shape[1])])