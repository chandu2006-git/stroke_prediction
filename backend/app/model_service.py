"""
Model service for StrokeGuard.

Loads the CatBoost model once at application startup and exposes
prediction + feature-schema helpers. The model path is configurable
via the STROKEGUARD_MODEL_PATH environment variable (default:
models/strokeguard_catboost.cbm).

No retraining happens here. The model file is expected to already
exist. If it is missing, ModelNotReady is raised and the API reports
a clear "model missing" status instead of fabricating predictions.
"""


from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool

logger = logging.getLogger("strokeguard.model")

# Directory this file lives in (backend/app). The project root is the
# parent's parent of that directory, so the default model path resolves to
#   <project_root>/models/strokeguard_catboost.cbm
# no matter which directory the server was started from.
_PACKAGE_DIR = Path(__file__).resolve().parent
# backend/app -> backend (the 'backend' package dir), then take the
# grandparent two levels up to reach the project root (stroke_prediction/).
_PROJECT_ROOT = _PACKAGE_DIR.parent.parent

# Exact feature order expected by the model. Must match training.
FEATURE_ORDER: List[str] = [
    "gender",
    "age",
    "hypertension",
    "heart_disease",
    "ever_married",
    "work_type",
    "Residence_type",
    "avg_glucose_level",
    "bmi",
    "bmi_missing",
    "smoking_status",
]

# Categorical features (by name). Indices are derived from FEATURE_ORDER.
CAT_FEATURE_NAMES: List[str] = [
    "gender",
    "ever_married",
    "work_type",
    "Residence_type",
    "smoking_status",
]

CAT_FEATURE_INDICES: List[int] = [FEATURE_ORDER.index(c) for c in CAT_FEATURE_NAMES]

# Numeric bounds documentation (for reference / frontend validation).
NUMERIC_BOUNDS = {
    "age": (1, 120),
    "avg_glucose_level": (0, 500),
    "bmi": (10, 70),
}

# Default model path. Resolved relative to the project root so the path is
# stable regardless of the caller's working directory.
DEFAULT_MODEL_PATH = _PROJECT_ROOT / "models" / "strokeguard_catboost.cbm"


class ModelNotReady(Exception):
    """Raised when the model file is missing or cannot be loaded."""


# Schema-friendly snake_case -> exact model training feature name.
# The model was trained with feature names like "Residence_type"; the API
# schema uses Pythonic snake_case ("residence_type"). Map them so the
# DataFrame is built with the exact training feature names.
SCHEMA_TO_FEATURE = {
    "residence_type": "Residence_type",
}



class ModelNotReady(Exception):
    """Raised when the model file is missing or cannot be loaded."""


# Schema-friendly snake_case -> exact model training feature name.
# The model was trained with feature names like "Residence_type"; the API
# schema uses Pythonic snake_case ("residence_type"). Map them so the
# DataFrame is built with the exact training feature names.
SCHEMA_TO_FEATURE = {
    "residence_type": "Residence_type",
}


class ModelService:
    """Wrapper around the loaded CatBoost model."""


    def __init__(self, model_path: Optional[str] = None) -> None:
        """
        Initialise the model service.

        The model path is resolved to an absolute Path:
          * If ``model_path`` is given, it is used (expanded and normalised).
          * Otherwise the ``STROKEGUARD_MODEL_PATH`` environment variable
            is read, if set.
          * Otherwise the project-root default is used:
            <project_root>/models/strokeguard_catboost.cbm

        Because the default is resolved with :class:`pathlib.Path`, it does
        not depend on the process working directory. The reported path is
        always the absolute path to the actual .cbm file.
        """
        if model_path is not None:
            self.model_path = Path(model_path).expanduser()
        else:
            env_path = os.environ.get("STROKEGUARD_MODEL_PATH")
            self.model_path = Path(env_path).expanduser() if env_path else DEFAULT_MODEL_PATH
        self.model: Optional[CatBoostClassifier] = None
        self.loaded: bool = False
        self.load_error: Optional[str] = None

    def _resolve_path(self) -> Path:
        """Return an absolute, normalised path for the model file."""
        return self.model_path.resolve()

    def load(self) -> None:
        """Load the model once. Safe to call at startup."""
        model_file = self._resolve_path()

        if not model_file.is_file():
            msg = (
                f"Model file not found at configured path: {model_file}. "
                "Place the trained CatBoost model (.cbm) there and restart. "
                "Prediction endpoint will return an error until the model exists."
            )
            self.load_error = msg
            self.loaded = False
            logger.error(msg)
            raise ModelNotReady(msg)

        try:
            model = CatBoostClassifier()
            model.load_model(str(model_file))
            self.model = model
            self.loaded = True
            self.load_error = None
            logger.info(f"CatBoost model loaded from: {model_file}")
        except Exception as exc:  # pragma: no cover - defensive
            self.load_error = f"Failed to load model: {exc}"
            self.loaded = False
            logger.exception(self.load_error)
            raise ModelNotReady(self.load_error) from exc

    def is_ready(self) -> bool:
        return self.loaded and self.model is not None

    def build_dataframe(self, payload: Dict[str, Any]) -> pd.DataFrame:
        """
        Construct a single-row DataFrame using exact training feature names
        and order. Categorical values are preserved as strings. Missing BMI
        is represented as NaN; bmi_missing is derived by the caller/schema.
        """
        row = {}
        for feature in FEATURE_ORDER:
            # Accept either the exact training feature name or the schema
            # snake_case alias (e.g. "residence_type" -> "Residence_type").
            alias = next(
                (k for k, v in SCHEMA_TO_FEATURE.items() if v == feature), feature
            )
            row[feature] = payload.get(feature, payload.get(alias))
        df = pd.DataFrame([row], columns=FEATURE_ORDER)
        return df

    def predict(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a prediction. Returns prediction label, positive-class
        probability, and the raw input used (for patient summary).
        """
        if not self.is_ready():
            raise ModelNotReady(self.load_error or "Model not loaded")

        df = self.build_dataframe(payload)
        pool = Pool(df, cat_features=CAT_FEATURE_INDICES)

        # CatBoost predict_proba returns shape (n_rows, n_classes)
        proba = self.model.predict_proba(pool)
        positive_class_probability = float(proba[0][1])
        prediction = int(self.model.predict(pool)[0])

        prediction_label = "Stroke" if prediction == 1 else "No Stroke"

        return {
            "prediction": prediction,
            "prediction_label": prediction_label,
            "probability": positive_class_probability,
            "input_dataframe": df,
        }

    def get_feature_order(self) -> List[str]:
        return list(FEATURE_ORDER)

    def get_cat_features(self) -> List[str]:
        return list(CAT_FEATURE_NAMES)