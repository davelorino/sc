from __future__ import annotations
import pandas as pd

def predict_weekly_uplift(pipe, feature_frame_like: pd.DataFrame, scenario: dict) -> float:
    """
    Predict expected weekly uplift for a scenario using the weekly within-campaign model.
    scenario keys should match features used in training (see Xcols in weekly_within_campaign.py).
    """
    dose_cols = [c for c in feature_frame_like.columns if c.startswith("dose_type::")]
    base_cols = ["n_assets_cov","n_types","disc_depth_bc","sov","esov","is_leader",
                 "running_carto_media","woy_sin","woy_cos"]
    Xcols = base_cols + dose_cols
    x = pd.DataFrame([{c: scenario.get(c, 0.0) for c in Xcols}])
    return float(pipe.predict(x)[0])

def predict_campaign_total(model, campaign_agg_like: pd.DataFrame, scenario: dict) -> float:
    """
    Predict expected campaign total uplift using the campaign-level model.
    scenario keys should match Xcols in campaign_level.py.
    """
    dose_cols = [c for c in campaign_agg_like.columns if c.startswith("dose_type::") and c.endswith("_avg")]
    Xcols = ["duration_weeks","n_assets_total","n_types_avg","esov_avg","disc_depth_avg","is_leader"] + dose_cols
    x = pd.DataFrame([{c: scenario.get(c, 0.0) for c in Xcols}])
    return float(model.predict(x.values)[0])
