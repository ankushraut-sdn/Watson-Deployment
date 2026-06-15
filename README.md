# IBM WATSON MACHINE LEARNING
## Benefits Eligibility Model
### Technical Specification & Model Documentation

**Version 1.0.0**  •  **March 28, 2026**  •  **HR Analytics Team**

---

## 1. Project Overview

This document describes all files required to deploy the Benefits Eligibility Classifier to IBM Watson Machine Learning. The model predicts whether an employee qualifies for company benefits based on their employment profile.

| Attribute | Value |
|---|---|
| Use Case | Employee benefits eligibility classification |
| Algorithm | Gradient Boosting Classifier (scikit-learn) |
| Owner | HR Analytics Team |
| Version | 1.0.0 |
| Risk Level | Medium — affects employee compensation |
| Training Date | 2026-03-28 |
| Output Classes | `not_eligible` \| `eligible` |

## 2. Repository & Artifact Structure

All files must be arranged exactly as shown. The `model/` directory is compressed into `model.tar.gz`, uploaded to IBM Cloud Object Storage (COS), and registered via the Watson ML Python client.

```text
benefits-model-watson/
├── model.tar.gz            ← upload to COS; register with Watson ML client
├── model/
│   ├── model.pkl               ← trained GradientBoostingClassifier
│   ├── preprocessor.joblib     ← fitted ColumnTransformer
│   └── label_map.json          ← {"0": "not_eligible", "1": "eligible"}
├── code/
│   ├── score.py                ← Watson ML entry-point (score function)
│   ├── requirements.txt        ← runtime Python dependencies
│   └── config.yaml             ← (optional) deployment config
├── metadata/
│   └── model_metadata.json     ← governance & audit fields
├── test/
│   ├── sample_request.json     ← fields/values columnar payload
│   └── expected_response.json  ← expected inference output
└── README.md
```

## 3. Artifact 1 — Model Bundle (MANDATORY)

### 3.1 model.tar.gz contents

1. `model.pkl` — serialised `GradientBoostingClassifier`
2. `preprocessor.joblib` — fitted `ColumnTransformer` (`StandardScaler` + `OneHotEncoder`)
3. `label_map.json` — integer-to-class-name mapping

After creating `model.tar.gz`, upload to COS and register using the Watson ML Python client:

```python
from ibm_watson_machine_learning import APIClient

client = APIClient(wml_credentials)
client.set.default_space(space_id)

# Store the tar.gz in COS
model_asset = client.repository.store_model(
    model         = 'model.tar.gz',
    meta_props    = {
        client.repository.ModelMetaNames.NAME:           'benefits-eligibility-v1',
        client.repository.ModelMetaNames.TYPE:           'scikit-learn_1.3',
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:
            client.software_specifications.get_uid_by_name('runtime-23.1-py3.10'),
    }
)
```

### 3.2 Feature Schema

| Feature | Type | Description / Allowed Values |
|---|---|---|
| age | int | Employee age in years (22–65) |
| employment_type | string | `full_time` \| `part_time` \| `contractor` |
| years_of_service | int | Completed years at the company (0–40) |
| weekly_hours | int | Contracted weekly hours (10–55) |
| department | string | `engineering` \| `sales` \| `hr` \| `finance` \| `operations` |
| salary_band | string | `band_1` \| `band_2` \| `band_3` \| `band_4` \| `band_5` |
| has_dependents | int | Binary flag — `0 = no dependents`, `1 = has dependents` |

### 3.3 label_map.json

```json
{
 "0": "not_eligible",
 "1": "eligible"
}
```

## 4. Artifact 2 — Inference Script (MANDATORY)

**File:** `code/score.py` — defines the single `score(payload)` function required by Watson ML. Unlike other platforms, Watson ML uses a columnar `fields/values` format rather than a list of record dicts.

### 4.1 Artifact loading — _load_artifacts()

Watson ML sets the `WATSON_ML_MODEL_DIR` environment variable to the directory containing the unpacked artifacts at runtime. The helper is called lazily on first invocation of `score()`.

```python
def _load_artifacts():
    global _model, _preprocessor, _label_map
    if _model is not None:
        return  # already loaded
    model_dir     = os.getenv('WATSON_ML_MODEL_DIR', '.')
    _model        = joblib.load(os.path.join(model_dir, 'model.pkl'))
    _preprocessor = joblib.load(os.path.join(model_dir, 'preprocessor.joblib'))
    with open(os.path.join(model_dir, 'label_map.json')) as f:
        _label_map = json.load(f)
```

### 4.2 score(payload) — entry-point

Called for every inference request. Accepts one or more `input_data` blocks, each with parallel `fields` and `values` lists. Builds a DataFrame, validates features, and returns results in the same columnar format.

```python
def score(payload: dict) -> dict:
    _load_artifacts()
    input_data = payload.get('input_data', [])
    frames = []
    for block in input_data:
        fields = block.get('fields', [])
        values = block.get('values', [])
        frames.append(pd.DataFrame(values, columns=fields))
    df          = pd.concat(frames, ignore_index=True)[EXPECTED_FEATURES]
    X           = _preprocessor.transform(df)
    raw_preds   = _model.predict(X).tolist()
    class_proba = _model.predict_proba(X).tolist()
    labeled     = [_label_map[str(p)] for p in raw_preds]
    label_order = [_label_map[k] for k in sorted(_label_map)]
    return {'predictions': [{
        'fields': ['prediction', 'probability', 'label_order'],
        'values': [[lbl, prob, label_order]
                   for lbl, prob in zip(labeled, class_proba)]
    }]}
```

## 5. Artifact 3 — Requirements File (MANDATORY)

**File:** `code/requirements.txt` — installed into the Watson ML runtime environment.

```text
numpy>=1.24
pandas>=2.0
scikit-learn>=1.3
xgboost>=2.0
joblib>=1.3
ibm-watson-machine-learning>=1.0.312
```

## 6. Artifact 4 — Model Metadata

**File:** `metadata/model_metadata.json` — used for the Watson ML model registry, governance, and auditing.

| Field | Value for This Model |
|---|---|
| model_version | 1.0.0 |
| owner | HR Analytics Team |
| use_case | Employee benefits eligibility classification |
| risk_level | medium |
| protected_attributes | age, has_dependents |
| training_date | 2026-03-28 |
| algorithm | GradientBoostingClassifier |
| training_samples | 1,600 |
| test_samples | 400 |

```json
{
  "model_version": "1.0.0",
  "owner": "HR Analytics Team",
  "use_case": "Employee benefits eligibility classification",
  "risk_level": "medium",
  "protected_attributes": ["age", "has_dependents"],
  "training_date": "2026-03-28",
  "algorithm": "GradientBoostingClassifier",
  "hyperparameters": {
    "n_estimators": 200, "learning_rate": 0.05,
    "max_depth": 4, "subsample": 0.8
  }
}
```

## 7. Artifact 5 — Sample Payloads (Validation)

### 7.1 Key difference — columnar fields/values format

Watson ML does not use a list-of-dicts format. Instead, a single block carries column names under `fields` and rows of values under `values`. This is the format produced by `client.deployments.score()` and expected by `score()`.

### 7.2 test/sample_request.json

```json
{
  "input_data": [
    {
      "fields": ["age","employment_type","years_of_service",
                 "weekly_hours","department","salary_band","has_dependents"],
      "values": [
        [34, "full_time",  5, 40, "engineering", "band_3", 1],
        [27, "part_time",  0, 20, "sales",       "band_1", 0]
      ]
    }
  ]
}
```

### 7.3 test/expected_response.json

```json
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
```

### 7.4 Invoking with the Watson ML Python client

```python
scoring_payload = {
    'input_data': [{
        'fields': ['age','employment_type','years_of_service',
                   'weekly_hours','department','salary_band','has_dependents'],
        'values': [[34, 'full_time', 5, 40, 'engineering', 'band_3', 1]]
    }]
}

result = client.deployments.score(deployment_uid, scoring_payload)
print(result['predictions'][0]['values'])
```

## 8. Deployment Notes

1. Create an IBM Cloud Object Storage (COS) bucket in the same region as your Watson ML instance.
2. Upload `model.tar.gz` to COS using the `ibm-watson-machine-learning` client or the IBM Cloud CLI.
3. Register the model using `client.repository.store_model()`, specifying type `scikit-learn_1.3` and the matching software spec (`runtime-23.1-py3.10` or later).
4. Create a deployment space in Watson ML Studio, then create an online deployment pointing to the registered model asset.
5. Score via `client.deployments.score(deployment_uid, payload)` using the `fields/values` columnar envelope — not a list of dicts.
6. Monitor `protected_attributes` (`age`, `has_dependents`) for bias drift using IBM OpenScale (Watson OpenPages) or the built-in Watson ML fairness monitor.

---

End of Document
