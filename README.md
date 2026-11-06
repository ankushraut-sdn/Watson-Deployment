# IBM Watson Machine Learning — Benefits Eligibility Model

## 1. Project Overview

**Project Name:** Watson-Deployment (Benefits Eligibility Model)
**Project Type:** ML model deployment (IBM Watson Machine Learning)
**Status:** Development

### Description

This project packages and deploys a Benefits Eligibility Classifier to IBM Watson Machine Learning. The model predicts whether an employee qualifies for company benefits (`eligible` / `not_eligible`) based on 7 employment-profile features, using a scikit-learn `GradientBoostingClassifier`.

## 2. Core Features

- Gradient Boosting classifier trained on 7 employment features (age, employment type, years of service, weekly hours, department, salary band, dependents) to predict benefits eligibility.
- Watson ML-compliant `watson_score.py` implementing the single `score(payload)` entry point.
- Unlike other cloud platforms, Watson ML uses a columnar `fields`/`values` request and response format rather than a list-of-dicts envelope; `score()` handles this format directly.
- Governance metadata tracking owner, risk level, protected attributes (`age`, `has_dependents`), training date, and hyperparameters, intended for bias monitoring via IBM OpenScale or the built-in Watson ML fairness monitor.

## 3. Tech Stack

| Component | Technology |
|---|---|
| Model | scikit-learn `GradientBoostingClassifier`, XGBoost |
| Data handling | pandas, numpy, joblib |
| Platform | IBM Watson Machine Learning (`ibm-watson-machine-learning` client, IBM Cloud Object Storage) |

## 4. Project Structure

```
Watson-Deployment/
├── train_updated.py          # model training script
├── watson_score.py           # Watson ML entry-point script (score function, fields/values format)
├── doc/                      # architecture diagram
└── LICENSE
```

## 5. Prerequisites

- An IBM Cloud account with a Watson Machine Learning instance and a Cloud Object Storage (COS) bucket
- `ibm-watson-machine-learning` Python client installed
- No `requirements.txt` is included in this repository — install scikit-learn, xgboost, pandas, numpy, joblib, and `ibm-watson-machine-learning` manually

## 6. How to Run

### Train

```bash
python train_updated.py
```

This produces `model.pkl`, `preprocessor.joblib`, and `label_map.json`.

### Deploy

1. Compress the `model/` directory into `model.tar.gz` and upload it to an IBM Cloud Object Storage (COS) bucket in the same region as your Watson ML instance.
2. Register the model using the Watson ML Python client:

```python
from ibm_watson_machine_learning import APIClient

client = APIClient(wml_credentials)
client.set.default_space(space_id)

model_asset = client.repository.store_model(
    model='model.tar.gz',
    meta_props={
        client.repository.ModelMetaNames.NAME: 'benefits-eligibility-v1',
        client.repository.ModelMetaNames.TYPE: 'scikit-learn_1.3',
        client.repository.ModelMetaNames.SOFTWARE_SPEC_UID:
            client.software_specifications.get_uid_by_name('runtime-23.1-py3.10'),
    }
)
```

3. Create a deployment space and an online deployment pointing to the registered model.
4. Score requests using the columnar envelope, not a list of dicts:

```python
scoring_payload = {
    'input_data': [{
        'fields': ['age', 'employment_type', 'years_of_service',
                   'weekly_hours', 'department', 'salary_band', 'has_dependents'],
        'values': [[34, 'full_time', 5, 40, 'engineering', 'band_3', 1]]
    }]
}
result = client.deployments.score(deployment_uid, scoring_payload)
```

5. Monitor `age` and `has_dependents` (protected attributes) for bias drift using IBM OpenScale or the Watson ML fairness monitor.

### Feature schema

| Feature | Type | Allowed values |
| :--- | :--- | :--- |
| age | int | 22-65 |
| employment_type | string | `full_time`, `part_time`, `contractor` |
| years_of_service | int | 0-40 |
| weekly_hours | int | 10-55 |
| department | string | `engineering`, `sales`, `hr`, `finance`, `operations` |
| salary_band | string | `band_1`...`band_5` |
| has_dependents | int | 0 or 1 |

## 7. Known Limitations

- No `requirements.txt` or automated test payloads are included in this repository, unlike the AWS variant.
- Bias/drift monitoring on protected attributes is a manual OpenScale/fairness-monitor setup step, not automated in this codebase.

## 8. License

MIT License (see `LICENSE`).
