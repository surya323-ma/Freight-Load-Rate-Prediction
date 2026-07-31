"""Feature engineering shared by training and inference."""
from __future__ import annotations

import numpy as np
import pandas as pd

CATEGORICAL_COLUMNS = ["pickup", "delivery", "equipment", "lane"]


def _haversine_miles(lat1, lon1, lat2, lon2) -> np.ndarray:
    r = 3958.8
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


class FeatureBuilder:
    """Learns imputation stats on training data, applies them everywhere else."""

    def __init__(self) -> None:
        self.weight_median_by_equipment_: dict[str, float] = {}
        self.weight_global_median_: float = np.nan
        self.market_index_median_: float = np.nan
        self.category_maps_: dict[str, dict[str, int]] = {}

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        clean_weight = df["weight"].abs()
        self.weight_global_median_ = float(clean_weight.median())
        self.weight_median_by_equipment_ = (
            clean_weight.groupby(df["equipment"]).median().to_dict()
        )
        self.market_index_median_ = float(df["market_index"].median())

        lanes = df["pickup"].astype(str) + "__" + df["delivery"].astype(str)
        for col, values in [
            ("pickup", df["pickup"]),
            ("delivery", df["delivery"]),
            ("equipment", df["equipment"]),
            ("lane", lanes),
        ]:
            categories = sorted(values.astype(str).unique().tolist())
            self.category_maps_[col] = {c: i for i, c in enumerate(categories)}
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()

        # --- weight: fix sign errors, impute missing ---
        out["weight_clean"] = out["weight"].abs()
        equip_median = out["equipment"].map(self.weight_median_by_equipment_)
        out["weight_clean"] = out["weight_clean"].fillna(equip_median)
        out["weight_clean"] = out["weight_clean"].fillna(self.weight_global_median_)
        out["weight_was_missing"] = df["weight"].isna().astype(int)
        out["weight_was_negative"] = (df["weight"] < 0).astype(int)

        # --- market_index: impute missing, flag ---
        out["market_index_clean"] = out["market_index"].fillna(self.market_index_median_)
        out["market_index_was_missing"] = out["market_index"].isna().astype(int)

        # --- date decomposition ---
        dt = pd.to_datetime(out["date"])
        out["month"] = dt.dt.month
        out["day_of_week"] = dt.dt.dayofweek
        out["day_of_year"] = dt.dt.dayofyear
        out["week_of_year"] = dt.dt.isocalendar().week.astype(int)
        out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)
        out["doy_sin"] = np.sin(2 * np.pi * out["day_of_year"] / 365.25)
        out["doy_cos"] = np.cos(2 * np.pi * out["day_of_year"] / 365.25)
        out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
        out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)

        # --- geo / distance features ---
        out["haversine_miles"] = _haversine_miles(
            out["pickup_lat"], out["pickup_lon"], out["delivery_lat"], out["delivery_lon"]
        )
        out["distance_vs_haversine"] = out["distance"] - out["haversine_miles"]
        out["log_distance"] = np.log1p(out["distance"])
        out["weight_per_distance"] = out["weight_clean"] / out["distance"].clip(lower=1)

        # --- categoricals -> integer codes (unseen -> -1) ---
        lanes = out["pickup"].astype(str) + "__" + out["delivery"].astype(str)
        out["lane"] = lanes
        for col in CATEGORICAL_COLUMNS:
            mapping = self.category_maps_[col]
            out[col + "_code"] = out[col].astype(str).map(mapping).fillna(-1).astype(int)

        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)


FEATURE_COLUMNS = [
    "distance",
    "log_distance",
    "haversine_miles",
    "distance_vs_haversine",
    "weight_clean",
    "weight_was_missing",
    "weight_was_negative",
    "weight_per_distance",
    "market_index_clean",
    "market_index_was_missing",
    "quote_signal",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
    "month",
    "day_of_week",
    "day_of_year",
    "week_of_year",
    "is_weekend",
    "doy_sin",
    "doy_cos",
    "dow_sin",
    "dow_cos",
    "pickup_code",
    "delivery_code",
    "equipment_code",
    "lane_code",
]

CATEGORICAL_FEATURE_NAMES = ["pickup_code", "delivery_code", "equipment_code", "lane_code"]
