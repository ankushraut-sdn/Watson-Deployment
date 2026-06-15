"""
IBM Watson Machine Learning Inference Script — Benefits Eligibility Classifier
==============================================================================
Implements the Watson ML custom deployment scoring contract.

Watson ML calls score() for every prediction request. The function receives
a dict with an "input_data" key containing a list of payload objects, each
with "fields" (column names) and "values" (rows of data).

Watson ML request envelope
--------------------------
{
  "input_data": [
    {
      "fields": ["age", "employment_type", ...],
      "values": [[34, "full_time", ...], [27, "part_time", ...]]
    }
  ]
}

Watson ML response envelope
---------------------------
{
  "predictions": [
    {
      "fields": ["prediction", "probability", "label_order"],
      "values": [
          ["eligible",     [0.08, 0.92], ["not_eligible", "eligible"]],
          ["not_eligible", [0.87, 0.13], ["not_eligible", "eligible"]]
      ]
    }
  ]
}
"""

import json
import logging
import os

import joblib
import pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EXPECTED_FEATURES = [
    "age",
    "employment_type",        # "full_time" | "part_time" | "contractor"
    "years_of_service",
    "weekly_hours",
    "department",
    "salary_band",            # "band_1" … "band_5"
    "has_dependents",         # 0 or 1
]

# Module-level globals — loaded once at container startup
_model        = None
_preprocessor = None
_label_map    = None


def _load_artifacts() -> None:
    """
    Lazy-load model artifacts on first call to score().
    Watson ML sets the WATSON_ML_MODEL_DIR environment variable to the
    directory containing the registered model files.  Falls back to '.'
    for local testing.
    """
    global _model, _preprocessor, _label_map

    if _model is not None:
        return  # already loaded

    model_dir = os.getenv("WATSON_ML_MODEL_DIR", ".")
    logger.info("Loading Watson ML artifacts from: %s", model_dir)

    _model        = joblib.load(os.path.join(model_dir, "model.pkl"))
    _preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))

    with open(os.path.join(model_dir, "label_map.json")) as f:
        _label_map = json.load(f)

    logger.info("Artifacts loaded. Classes: %s", list(_label_map.values()))


# ---------------------------------------------------------------------------
# score() — Watson ML entry-point
# ---------------------------------------------------------------------------

def score(payload: dict) -> dict:
    """
    Watson ML scoring entry-point.

    Parameters
    ----------
    payload : dict
        Watson ML request body.  Must contain "input_data" — a list of
        payload objects each with "fields" and "values".

    Returns
    -------
    dict
        Watson ML response body with "predictions" list.

    Raises
    ------
    ValueError
        If required features are missing or the payload is malformed.
    """
    _load_artifacts()

    try:
        input_data = payload.get("input_data", [])
        if not input_data:
            raise ValueError("'input_data' key is missing or empty.")

        # Support one or more payload objects; concatenate into a single df
        frames = []
        for block in input_data:
            fields = block.get("fields", [])
            values = block.get("values", [])

            if not fields or not values:
                raise ValueError("Each input_data block must have 'fields' and 'values'.")

            frames.append(pd.DataFrame(values, columns=fields))

        df = pd.concat(frames, ignore_index=True)

        missing = set(EXPECTED_FEATURES) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")

        df = df[EXPECTED_FEATURES]

        # Inference
        X           = _preprocessor.transform(df)
        raw_preds   = _model.predict(X).tolist()
        class_proba = _model.predict_proba(X).tolist()

        label_order   = [_label_map[k] for k in sorted(_label_map)]
        labeled_preds = [_label_map[str(p)] for p in raw_preds]

        # Watson ML response: parallel "fields" + "values" structure
        return {
            "predictions": [
                {
                    "fields": ["prediction", "probability", "label_order"],
                    "values": [
                        [label, prob, label_order]
                        for label, prob in zip(labeled_preds, class_proba)
                    ],
                }
            ]
        }

    except Exception as exc:
        logger.error("Scoring error: %s", exc, exc_info=True)
        # Watson ML surfaces errors via a "predictions" block with an error field
        return {
            "predictions": [
                {
                    "fields": ["error"],
                    "values": [[str(exc)]],
                }
            ]
        }
