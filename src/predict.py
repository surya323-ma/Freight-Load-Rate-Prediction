"""Score data/validation.csv and data/december_chart_inputs.csv with the final model."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import FEATURE_COLUMNS

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REPORTS = ROOT / "reports"

# Fixed lat/lon for the December fixed-lane scenario (looked up from the
# labeled data, where every city's coordinates are constant).
LEXINGTON = (36.99152, -84.99876)
FORT_WAYNE = (41.31561, -85.36206)

# december_chart_inputs.csv only varies `date`; it has no market_index /
# quote_signal columns. Those two signals are noisy month-to-month with no
# clear trend (checked during EDA), so the most defensible estimate for a
# near-future window is the mean of the most recent labeled month (October).
market_index_default = 0.957502
quote_signal_default = 1.942758



def main() -> None:
    bundle = joblib.load(REPORTS / "final_model.joblib")
    model, builder = bundle["model"], bundle["builder"]

    # --- validation_predictions.csv ---
    validation = pd.read_csv(DATA / "validation.csv")
    template = pd.read_csv(DATA / "validation_predictions_template.csv")

    feat = builder.transform(validation)
    pred = np.expm1(model.predict(feat[FEATURE_COLUMNS]))
    pred = np.clip(pred, 1.0, None)  # guard against non-positive predictions

    out = template[["load_id"]].merge(
        pd.DataFrame({"load_id": validation["load_id"], "predicted_rate": pred}),
        on="load_id",
        how="left",
    )
    assert out["predicted_rate"].isna().sum() == 0, "missing predictions for some load_id"
    assert list(out.columns) == ["load_id", "predicted_rate"]
    assert len(out) == len(template)
    out_path = ROOT / "validation_predictions.csv"
    out.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(out)} rows)")

    # --- december_chart_inputs.csv (fixed lane, 31 daily predictions) ---
    december = pd.read_csv(DATA / "december_chart_inputs.csv")
    dec_input = december.drop(columns=["predicted_rate"]).copy()
    # december_chart_inputs.csv has no lat/lon/quote_signal/market_index columns,
    # so derive them the same way they're fixed for this lane in the labeled data.
    dec_input["pickup_lat"] = LEXINGTON[0]
    dec_input["pickup_lon"] = LEXINGTON[1]
    dec_input["delivery_lat"] = FORT_WAYNE[0]
    dec_input["delivery_lon"] = FORT_WAYNE[1]
    dec_input["market_index"] = market_index_default
    dec_input["quote_signal"] = quote_signal_default

    dec_feat = builder.transform(dec_input)
    dec_pred = np.expm1(model.predict(dec_feat[FEATURE_COLUMNS]))
    december["predicted_rate"] = np.clip(dec_pred, 1.0, None)
    dec_out_path = DATA / "december_chart_inputs.csv"
    december.to_csv(dec_out_path, index=False)
    print(f"Wrote {dec_out_path} (December predictions filled)")


if __name__ == "__main__":
    main()
