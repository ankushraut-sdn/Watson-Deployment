"""
train.py — Benefits Eligibility Classifier (multi-cloud)
=========================================================
Trains a GradientBoostingClassifier and packages artifacts for one or more
cloud ML platforms in a single run.

Usage
-----
# All platforms (including Watson ML):
    python train.py --platform all

# Single platform:
    python train.py --platform sagemaker
    python train.py --platform azure
    python train.py --platform vertex
    python train.py --platform watson

# Multiple platforms:
    python train.py --platform sagemaker vertex watson

# Override output root:
    python train.py --platform all --output-dir ./my-builds

# Skip model.tar.gz for SageMaker / Watson (local inspection):
    python train.py --platform sagemaker watson --no-tar

Platform output structure
-------------------------
SageMaker → <output_dir>/sagemaker/
                model.tar.gz
                model/model.pkl, preprocessor.joblib, label_map.json
                code/inference.py, requirements.txt
                metadata/model_metadata.json
                test/sample_request.json, expected_response.json

Azure     → <output_dir>/azure/
                model/model.pkl, preprocessor.joblib, label_map.json
                code/score.py, requirements.txt
                metadata/model_metadata.json
                test/sample_request.json  ← {"data": [...]}
                test/expected_response.json

Vertex    → <output_dir>/vertex/
                model/model.pkl, preprocessor.joblib, label_map.json
                code/predictor.py, requirements.txt
                metadata/model_metadata.json
                test/sample_request.json  ← {"instances": [...]}
                test/expected_response.json

Watson ML → <output_dir>/watson/
                model.tar.gz
                model/model.pkl, preprocessor.joblib, label_map.json
                code/score.py, requirements.txt
                metadata/model_metadata.json
                test/sample_request.json  ← {"input_data": [{"fields":[...], "values":[...]}]}
                test/expected_response.json
"""

import argparse
import json
import tarfile
from datetime import date
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Constants ────────────────────────────────────────────────────────────────

RANDOM_STATE = 42
PLATFORMS    = ("sagemaker", "azure", "vertex", "watson")

LABEL_MAP = {"0": "not_eligible", "1": "eligible"}

MODEL_METADATA = {
    "model_version":        "1.0.0",
    "owner":                "HR Analytics Team",
    "use_case":             "Employee benefits eligibility classification",
    "risk_level":           "medium",
    "protected_attributes": ["age", "has_dependents"],
    "training_date":        str(date.today()),
    "algorithm":            "GradientBoostingClassifier",
    "framework":            "scikit-learn",
    "input_features": [
        "age", "employment_type", "years_of_service",
        "weekly_hours", "department", "salary_band", "has_dependents",
    ],
    "output_classes": ["not_eligible", "eligible"],
    "hyperparameters": {
        "n_estimators":  200,
        "learning_rate": 0.05,
        "max_depth":     4,
        "subsample":     0.8,
    },
}

# Shared sample instances — envelope wrapping differs per platform
SAMPLE_INSTANCES = [
    {
        "age": 34, "employment_type": "full_time", "years_of_service": 5,
        "weekly_hours": 40, "department": "engineering",
        "salary_band": "band_3", "has_dependents": 1,
    },
    {
        "age": 27, "employment_type": "part_time", "years_of_service": 0,
        "weekly_hours": 20, "department": "sales",
        "salary_band": "band_1", "has_dependents": 0,
    },
]

EXPECTED_PREDICTIONS = ["eligible", "not_eligible"]
EXPECTED_PROBA       = [[0.08, 0.92], [0.87, 0.13]]

# Feature order used in Watson ML fields/values payloads
FEATURE_ORDER = [
    "age", "employment_type", "years_of_service",
    "weekly_hours", "department", "salary_band", "has_dependents",
]


# ════════════════════════════════════════════════════════════════════════════
# Step 1 — Train  (platform-agnostic)
# ════════════════════════════════════════════════════════════════════════════

def build_dataset(n: int = 2_000) -> tuple:
    """Generate synthetic employee dataset and binary eligibility labels."""
    np.random.seed(RANDOM_STATE)

    df = pd.DataFrame({
        "age":              np.random.randint(22, 65, n),
        "employment_type":  np.random.choice(
                                ["full_time", "part_time", "contractor"], n,
                                p=[0.6, 0.25, 0.15]),
        "years_of_service": np.clip(np.random.exponential(5, n), 0, 40).astype(int),
        "weekly_hours":     np.random.randint(10, 55, n),
        "department":       np.random.choice(
                                ["engineering", "sales", "hr", "finance", "operations"], n),
        "salary_band":      np.random.choice(
                                ["band_1", "band_2", "band_3", "band_4", "band_5"], n,
                                p=[0.15, 0.25, 0.30, 0.20, 0.10]),
        "has_dependents":   np.random.randint(0, 2, n),
    })

    eligible = (
        (df["employment_type"] == "full_time") &
        (df["weekly_hours"]     >= 30) &
        (df["years_of_service"] >= 1)
    ).astype(int)

    flip = np.random.rand(n) < 0.05          # 5 % label noise
    eligible[flip] = 1 - eligible[flip]

    return df, eligible


def train_model(df: pd.DataFrame, y: pd.Series) -> tuple:
    """
    Fit a ColumnTransformer + GradientBoostingClassifier.

    Returns
    -------
    (preprocessor, clf, classification_report_str, metadata_dict)
    """
    numeric_feats     = ["age", "years_of_service", "weekly_hours", "has_dependents"]
    categorical_feats = ["employment_type", "department", "salary_band"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(),                      numeric_feats),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_feats),
    ])

    clf = GradientBoostingClassifier(
        n_estimators=200, learning_rate=0.05,
        max_depth=4, subsample=0.8,
        random_state=RANDOM_STATE,
    )

    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y,
    )

    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t  = preprocessor.transform(X_test)
    clf.fit(X_train_t, y_train)

    report = classification_report(
        y_test, clf.predict(X_test_t),
        target_names=["not_eligible", "eligible"],
    )

    metadata = {
        **MODEL_METADATA,
        "training_samples": int(len(X_train)),
        "test_samples":     int(len(X_test)),
    }

    return preprocessor, clf, report, metadata


# ════════════════════════════════════════════════════════════════════════════
# Step 2 — Write shared artifacts into a staging directory
# ════════════════════════════════════════════════════════════════════════════

def write_shared_artifacts(stage_dir: Path, preprocessor, clf, metadata: dict) -> None:
    """
    Write files identical across all platforms:
        model/model.pkl
        model/preprocessor.joblib
        model/label_map.json
        metadata/model_metadata.json
        test/expected_response.json
    """
    (stage_dir / "model").mkdir(parents=True, exist_ok=True)
    (stage_dir / "metadata").mkdir(parents=True, exist_ok=True)
    (stage_dir / "test").mkdir(parents=True, exist_ok=True)

    joblib.dump(clf,          stage_dir / "model" / "model.pkl")
    joblib.dump(preprocessor, stage_dir / "model" / "preprocessor.joblib")

    _write_json(stage_dir / "model"    / "label_map.json",        LABEL_MAP)
    _write_json(stage_dir / "metadata" / "model_metadata.json",   metadata)
    _write_json(stage_dir / "test"     / "expected_response.json", {
        "predictions":   EXPECTED_PREDICTIONS,
        "probabilities": EXPECTED_PROBA,
        "label_order":   [LABEL_MAP[k] for k in sorted(LABEL_MAP)],
    })


# ════════════════════════════════════════════════════════════════════════════
# Step 3 — Platform packagers
# ════════════════════════════════════════════════════════════════════════════

class PlatformPackager:
    """
    Base packager.  Subclasses override:
        name              str   folder name under output root
        script_name()     str   inference file name
        inference_script() str  full file contents
        requirements()    str   requirements.txt contents
        sample_request()  dict  platform-specific request envelope
        post_process()          called after files are written (e.g. tar)
    """

    name: str = ""

    def __init__(self, root: Path, create_tar: bool = True):
        self.out_dir    = root / self.name
        self.create_tar = create_tar

    def package(self, preprocessor, clf, metadata: dict) -> None:
        print(f"\n{'─' * 55}")
        print(f"  Packaging → {self.name}")
        print(f"{'─' * 55}")

        self.out_dir.mkdir(parents=True, exist_ok=True)

        write_shared_artifacts(self.out_dir, preprocessor, clf, metadata)

        code_dir = self.out_dir / "code"
        code_dir.mkdir(exist_ok=True)
        (code_dir / self.script_name()).write_text(self.inference_script())
        (code_dir / "requirements.txt").write_text(self.requirements())

        _write_json(self.out_dir / "test" / "sample_request.json", self.sample_request())

        self.post_process()
        print(f"  Output : {self.out_dir.resolve()}")

    def script_name(self) -> str:          raise NotImplementedError
    def inference_script(self) -> str:     raise NotImplementedError
    def requirements(self) -> str:         raise NotImplementedError
    def sample_request(self) -> dict:      raise NotImplementedError
    def post_process(self) -> None:        pass


# ── SageMaker ─────────────────────────────────────────────────────────────────

class SageMakerPackager(PlatformPackager):
    name = "sagemaker"

    def script_name(self) -> str:
        return "inference.py"

    def sample_request(self) -> dict:
        return {"instances": SAMPLE_INSTANCES}

    def requirements(self) -> str:
        return (
            "numpy>=1.24\n"
            "pandas>=2.0\n"
            "scikit-learn>=1.3\n"
            "xgboost>=2.0\n"
            "joblib>=1.3\n"
        )

    def inference_script(self) -> str:
        return '''\
"""AWS SageMaker inference entry-point — four handler functions."""
import os, json, logging
import joblib, pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EXPECTED_FEATURES = [
    "age", "employment_type", "years_of_service",
    "weekly_hours", "department", "salary_band", "has_dependents",
]


def model_fn(model_dir):
    model        = joblib.load(os.path.join(model_dir, "model.pkl"))
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
    with open(os.path.join(model_dir, "label_map.json")) as f:
        label_map = json.load(f)
    logger.info("Model loaded from %s", model_dir)
    return {"model": model, "preprocessor": preprocessor, "label_map": label_map}


def input_fn(request_body, content_type="application/json"):
    if content_type == "application/json":
        data = json.loads(request_body)
        if isinstance(data, dict) and "instances" in data:
            data = data["instances"]
        if isinstance(data, dict):
            data = [data]
        df = pd.DataFrame(data)
    elif content_type == "text/csv":
        from io import StringIO
        df = pd.read_csv(StringIO(request_body))
    else:
        raise ValueError(f"Unsupported content type: {content_type}")
    missing = set(EXPECTED_FEATURES) - set(df.columns)
    if missing:
        raise ValueError(f"Missing features: {missing}")
    return df[EXPECTED_FEATURES]


def predict_fn(input_data, model_artifacts):
    X           = model_artifacts["preprocessor"].transform(input_data)
    raw_preds   = model_artifacts["model"].predict(X).tolist()
    class_proba = model_artifacts["model"].predict_proba(X).tolist()
    label_map   = model_artifacts["label_map"]
    return {
        "predictions":   [label_map[str(p)] for p in raw_preds],
        "probabilities": class_proba,
        "label_order":   [label_map[k] for k in sorted(label_map)],
    }


def output_fn(prediction, accept="application/json"):
    if accept == "application/json":
        return json.dumps(prediction), "application/json"
    raise ValueError(f"Unsupported accept type: {accept}")
'''

    def post_process(self) -> None:
        if not self.create_tar:
            return
        tar_path  = self.out_dir / "model.tar.gz"
        model_dir = self.out_dir / "model"
        with tarfile.open(tar_path, "w:gz") as tar:
            for f in model_dir.iterdir():
                tar.add(f, arcname=f.name)
        print(f"  Created: model.tar.gz  ({tar_path.stat().st_size // 1024} KB)")


# ── Azure ─────────────────────────────────────────────────────────────────────

class AzurePackager(PlatformPackager):
    name = "azure"

    def script_name(self) -> str:
        return "score.py"

    def sample_request(self) -> dict:
        return {"data": SAMPLE_INSTANCES}

    def requirements(self) -> str:
        return (
            "numpy>=1.24\n"
            "pandas>=2.0\n"
            "scikit-learn>=1.3\n"
            "xgboost>=2.0\n"
            "joblib>=1.3\n"
            "azureml-defaults\n"
        )

    def inference_script(self) -> str:
        return '''\
"""Azure ML inference entry-point — init() + run()."""
import os, json, logging
import joblib, pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

model = preprocessor = label_map = None

EXPECTED_FEATURES = [
    "age", "employment_type", "years_of_service",
    "weekly_hours", "department", "salary_band", "has_dependents",
]


def init():
    global model, preprocessor, label_map
    model_dir    = os.getenv("AZUREML_MODEL_DIR", ".")
    model        = joblib.load(os.path.join(model_dir, "model.pkl"))
    preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
    with open(os.path.join(model_dir, "label_map.json")) as f:
        label_map = json.load(f)
    logger.info("Model loaded from %s", model_dir)


def run(raw_data):
    try:
        payload = json.loads(raw_data)
        if isinstance(payload, dict) and "data" in payload:
            payload = payload["data"]
        if isinstance(payload, dict):
            payload = [payload]
        df = pd.DataFrame(payload)
        missing = set(EXPECTED_FEATURES) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        df = df[EXPECTED_FEATURES]
        X           = preprocessor.transform(df)
        raw_preds   = model.predict(X).tolist()
        class_proba = model.predict_proba(X).tolist()
        return json.dumps({
            "predictions":   [label_map[str(p)] for p in raw_preds],
            "probabilities": class_proba,
            "label_order":   [label_map[k] for k in sorted(label_map)],
        })
    except Exception as exc:
        logger.error("Inference error: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc)})
'''


# ── Vertex ─────────────────────────────────────────────────────────────────────

class VertexPackager(PlatformPackager):
    name = "vertex"

    def script_name(self) -> str:
        return "predictor.py"

    def sample_request(self) -> dict:
        return {"instances": SAMPLE_INSTANCES}

    def requirements(self) -> str:
        return (
            "numpy>=1.24\n"
            "pandas>=2.0\n"
            "scikit-learn>=1.3\n"
            "xgboost>=2.0\n"
            "joblib>=1.3\n"
        )

    def inference_script(self) -> str:
        return '''\
"""Google Vertex AI Custom Prediction Routine — Predictor class."""
import os, json, logging
import joblib, pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EXPECTED_FEATURES = [
    "age", "employment_type", "years_of_service",
    "weekly_hours", "department", "salary_band", "has_dependents",
]


class Predictor:
    def __init__(self):
        self.model = self.preprocessor = self.label_map = None

    def load(self, artifacts_uri: str):
        self.model        = joblib.load(os.path.join(artifacts_uri, "model.pkl"))
        self.preprocessor = joblib.load(os.path.join(artifacts_uri, "preprocessor.joblib"))
        with open(os.path.join(artifacts_uri, "label_map.json")) as f:
            self.label_map = json.load(f)
        logger.info("Model loaded from %s", artifacts_uri)

    def preprocess(self, prediction_input: dict) -> pd.DataFrame:
        instances = prediction_input.get("instances", prediction_input)
        if isinstance(instances, dict):
            instances = [instances]
        df = pd.DataFrame(instances)
        missing = set(EXPECTED_FEATURES) - set(df.columns)
        if missing:
            raise ValueError(f"Missing features: {missing}")
        return df[EXPECTED_FEATURES]

    def predict(self, instances: pd.DataFrame) -> dict:
        X = self.preprocessor.transform(instances)
        return {
            "raw_predictions": self.model.predict(X).tolist(),
            "probabilities":   self.model.predict_proba(X).tolist(),
        }

    def postprocess(self, prediction_results: dict) -> dict:
        label_order = [self.label_map[k] for k in sorted(self.label_map)]
        labeled     = [self.label_map[str(p)] for p in prediction_results["raw_predictions"]]
        return {"predictions": [
            {"label": lbl, "probability": max(prob),
             "all_proba": dict(zip(label_order, prob))}
            for lbl, prob in zip(labeled, prediction_results["probabilities"])
        ]}
'''


# ── Watson ML ─────────────────────────────────────────────────────────────────

class WatsonMLPackager(PlatformPackager):
    """
    IBM Watson Machine Learning packager.

    Key differences from the other platforms:
    - Entry-point is a single score(payload) function (no class, no init/run split)
    - Artifacts are bundled as model.tar.gz and uploaded to Cloud Object Storage (COS)
    - Request envelope uses fields/values columnar format instead of a list of dicts
    - Response envelope mirrors the same fields/values structure under "predictions"
    - WATSON_ML_MODEL_DIR env var points to the unpacked artifact directory at runtime
    - ibm-watson-machine-learning SDK is required in requirements.txt
    """

    name = "watson"

    def script_name(self) -> str:
        return "score.py"

    def sample_request(self) -> dict:
        """
        Watson ML uses a columnar fields/values format rather than a list of
        record dicts.  This mirrors how the Watson ML Python client sends data
        via client.deployments.score().
        """
        values = [
            [inst[f] for f in FEATURE_ORDER]
            for inst in SAMPLE_INSTANCES
        ]
        return {
            "input_data": [
                {
                    "fields": FEATURE_ORDER,
                    "values": values,
                }
            ]
        }

    def requirements(self) -> str:
        return (
            "numpy>=1.24\n"
            "pandas>=2.0\n"
            "scikit-learn>=1.3\n"
            "xgboost>=2.0\n"
            "joblib>=1.3\n"
            "ibm-watson-machine-learning>=1.0.312\n"   # Watson ML client + runtime
        )

    def inference_script(self) -> str:
        return '''\
"""
IBM Watson Machine Learning inference entry-point.

Watson ML calls score(payload) for every prediction request.

Request envelope
----------------
{
  "input_data": [
    {
      "fields": ["age", "employment_type", ...],
      "values": [[34, "full_time", ...], [27, "part_time", ...]]
    }
  ]
}

Response envelope
-----------------
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
import json, logging, os
import joblib, pandas as pd

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

EXPECTED_FEATURES = [
    "age", "employment_type", "years_of_service",
    "weekly_hours", "department", "salary_band", "has_dependents",
]

_model = _preprocessor = _label_map = None


def _load_artifacts():
    global _model, _preprocessor, _label_map
    if _model is not None:
        return
    model_dir     = os.getenv("WATSON_ML_MODEL_DIR", ".")
    _model        = joblib.load(os.path.join(model_dir, "model.pkl"))
    _preprocessor = joblib.load(os.path.join(model_dir, "preprocessor.joblib"))
    with open(os.path.join(model_dir, "label_map.json")) as f:
        _label_map = json.load(f)
    logger.info("Watson ML artifacts loaded from %s", model_dir)


def score(payload: dict) -> dict:
    _load_artifacts()
    try:
        input_data = payload.get("input_data", [])
        if not input_data:
            raise ValueError("\'input_data\' key is missing or empty.")

        frames = []
        for block in input_data:
            fields = block.get("fields", [])
            values = block.get("values", [])
            if not fields or not values:
                raise ValueError("Each input_data block must have \'fields\' and \'values\'.")
            frames.append(pd.DataFrame(values, columns=fields))

        df = pd.concat(frames, ignore_index=True)
        missing = set(EXPECTED_FEATURES) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required feature(s): {missing}")

        df = df[EXPECTED_FEATURES]
        X           = _preprocessor.transform(df)
        raw_preds   = _model.predict(X).tolist()
        class_proba = _model.predict_proba(X).tolist()

        label_order   = [_label_map[k] for k in sorted(_label_map)]
        labeled_preds = [_label_map[str(p)] for p in raw_preds]

        return {
            "predictions": [{
                "fields": ["prediction", "probability", "label_order"],
                "values": [
                    [label, prob, label_order]
                    for label, prob in zip(labeled_preds, class_proba)
                ],
            }]
        }

    except Exception as exc:
        logger.error("Scoring error: %s", exc, exc_info=True)
        return {"predictions": [{"fields": ["error"], "values": [[str(exc)]]}]}
'''

    def post_process(self) -> None:
        """
        Bundle model/ into model.tar.gz.
        Watson ML requires artifacts uploaded to Cloud Object Storage as a
        tar.gz archive before registering the model via the Python client.
        """
        if not self.create_tar:
            return
        tar_path  = self.out_dir / "model.tar.gz"
        model_dir = self.out_dir / "model"
        with tarfile.open(tar_path, "w:gz") as tar:
            for f in model_dir.iterdir():
                tar.add(f, arcname=f.name)
        print(f"  Created: model.tar.gz  ({tar_path.stat().st_size // 1024} KB)")


# ════════════════════════════════════════════════════════════════════════════
# Helpers
# ════════════════════════════════════════════════════════════════════════════

def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2))


_PACKAGER_MAP: dict[str, type] = {
    "sagemaker": SageMakerPackager,
    "azure":     AzurePackager,
    "vertex":    VertexPackager,
    "watson":    WatsonMLPackager,
}


# ════════════════════════════════════════════════════════════════════════════
# CLI
# ════════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the benefits model and package for cloud ML platforms.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python train.py --platform all\n"
            "  python train.py --platform sagemaker azure\n"
            "  python train.py --platform watson --output-dir ./watson-build\n"
            "  python train.py --platform sagemaker watson --no-tar\n"
        ),
    )
    parser.add_argument(
        "--platform", "-p",
        nargs="+",
        choices=[*PLATFORMS, "all"],
        default=["all"],
        metavar="PLATFORM",
        help="sagemaker | azure | vertex | watson | all  (default: all; multiple accepted)",
    )
    parser.add_argument(
        "--output-dir", "-o",
        default="./dist",
        help="Root output directory (default: ./dist)",
    )
    parser.add_argument(
        "--no-tar",
        action="store_true",
        help="Skip model.tar.gz for SageMaker and Watson (useful for local inspection)",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=2_000,
        help="Synthetic training samples (default: 2000)",
    )
    return parser.parse_args()


# ════════════════════════════════════════════════════════════════════════════
# Entry point
# ════════════════════════════════════════════════════════════════════════════

def main() -> None:
    args    = parse_args()
    targets = list(PLATFORMS) if "all" in args.platform else list(dict.fromkeys(args.platform))
    root    = Path(args.output_dir)

    print("=" * 55)
    print("  Benefits Eligibility Classifier — multi-cloud build")
    print("=" * 55)
    print(f"  Platforms  : {', '.join(targets)}")
    print(f"  Output dir : {root.resolve()}")
    print(f"  Samples    : {args.samples:,}")

    print("\n[1/2] Training model …")
    df, y                               = build_dataset(args.samples)
    preprocessor, clf, report, metadata = train_model(df, y)
    print(report)

    print("[2/2] Packaging artifacts …")
    for platform in targets:
        packager = _PACKAGER_MAP[platform](root, create_tar=not args.no_tar)
        packager.package(preprocessor, clf, metadata)

    print("\n" + "=" * 55)
    print("  Build complete.")
    print("=" * 55)


if __name__ == "__main__":
    main()
