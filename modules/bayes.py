import numpy as np

# Environmental risk factors
WEATHER_FACTOR = {"clear": 0.08, "night": 0.14, "rain": 0.20, "fog": 0.24}
SURFACE_FACTOR = {"dry": 0.07, "wet": 0.16, "icy": 0.25}

def prob_to_odds(p: float) -> float:
    """Convert probability to odds ratio."""
    p = float(np.clip(p, 1e-6, 1 - 1e-6))
    return p / (1 - p)

def odds_to_prob(o: float) -> float:
    """Convert odds ratio to probability."""
    return o / (1 + o)

def compute_base_risk(weather_condition: str, road_surface_condition: str, visibility_range_m: float) -> float:
    """Compute baseline risk based on environmental conditions."""
    return min(
        0.95,
        WEATHER_FACTOR.get(str(weather_condition).lower(), 0.1)
        + SURFACE_FACTOR.get(str(road_surface_condition).lower(), 0.1)
        + max(0.0, (200 - float(visibility_range_m)) / 1000.0),
    )

def apply_bayesian_update(risk_score_ml, weather_condition, road_surface_condition, visibility_range_m):
    """
    Update risk assessment using Bayesian inference (Log-Odds Update).
    
    P(risk | ML_obs) is updated from P(base_risk) using the Likelihood Ratio from the ML model.
    """
    p_base = compute_base_risk(weather_condition, road_surface_condition, visibility_range_m)
    
    # Likelihood ratio based on ML model output (High risk score > 0.5 increases total risk)
    lr = np.exp((float(risk_score_ml) - 0.5) * 3.0)
    p_risk_dynamic = float(odds_to_prob(prob_to_odds(p_base) * lr))
    
    return p_base, p_risk_dynamic

