"""
Explanation service for StrokeGuard.

Generates SHAP (feature-level) and LIME (local surrogate)
explanations for a single patient prediction.

Only explanations that are actually computed are returned. If SHAP
or LIME cannot be produced, the corresponding block reports
available=False with a graceful message - no fabricated explanations.
"""

import logging
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
import shap

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except Exception:  # pragma: no cover
    LIME_AVAILABLE = False

logger = logging.getLogger("strokeguard.explanation")

from .model_service import FEATURE_ORDER, CAT_FEATURE_NAMES, CAT_FEATURE_INDICES


class ExplanationService:
    def __init__(
        self,
        model: CatBoostClassifier,
        background_data: Optional[pd.DataFrame] = None,
    ) -> None:
        self.model = model
        self.background_data = background_data
        self._shap_explainer: Optional[shap.Explainer] = None
        self._lime_explainer: Optional[LimeTabularExplainer] = None
        self._lime_mapping: Optional[Dict[str, Dict[str, int]]] = None
        self._init_shap()
        self._init_lime()

    # ------------------------------------------------------------------ SHAP
    def _init_shap(self) -> None:
        try:
            self._shap_explainer = shap.Explainer(self.model)
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"SHAP explainer could not be initialized: {exc}")
            self._shap_explainer = None

    def shap_explain(self, df: pd.DataFrame) -> Dict[str, Any]:
        if self._shap_explainer is None:
            return {
                "available": False,
                "message": "SHAP explanation is not available.",
                "contributions": [],
                "base_value": None,
            }
        try:
            pool = Pool(df, cat_features=CAT_FEATURE_INDICES)
            shap_values = self._shap_explainer(pool)

            values = np.array(shap_values.values)
            if values.ndim == 2:
                row_values = values[0]
            else:
                row_values = values

            base_values = np.array(shap_values.base_values)
            base_value = float(base_values[0]) if base_values.ndim > 0 else float(base_values)

            contributions: List[Dict[str, Any]] = []
            for name, val in zip(FEATURE_ORDER, row_values):
                v = float(val)
                contributions.append({
                    "feature": name,
                    "value": round(v, 6),
                    "direction": "positive" if v > 0 else ("negative" if v < 0 else "neutral"),
                })

            contributions.sort(key=lambda c: abs(c["value"]), reverse=True)

            return {
                "available": True,
                "message": "SHAP feature-level contributions.",
                "contributions": contributions,
                "base_value": round(base_value, 6),
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("SHAP explanation failed")
            return {
                "available": False,
                "message": f"SHAP explanation could not be computed: {exc}",
                "contributions": [],
                "base_value": None,
            }

    # ------------------------------------------------------------------ LIME
    def _init_lime(self) -> None:
        if not LIME_AVAILABLE or self.background_data is None:
            logger.info("LIME explainer not initialized (background data missing or lime unavailable).")
            return

        try:
            train_enc = self.background_data.copy()
            mapping: Dict[str, Dict[str, int]] = {}
            for c in CAT_FEATURE_NAMES:
                uniq = sorted(train_enc[c].astype(str).unique().tolist())
                mapping[c] = {v: i for i, v in enumerate(uniq)}
                train_enc[c] = train_enc[c].astype(str).map(mapping[c])

            categorical_names = {}
            for i in CAT_FEATURE_INDICES:
                c = FEATURE_ORDER[i]
                categorical_names[i] = sorted(
                    self.background_data[c].astype(str).unique().tolist()
                )

            def predict_fn(data):
                data = pd.DataFrame(data, columns=FEATURE_ORDER)
                data_dec = data.copy()
                for c in CAT_FEATURE_NAMES:
                    inv = {i: v for v, i in mapping[c].items()}
                    data_dec[c] = data[c].astype(float).map(inv)
                pool = Pool(data_dec, cat_features=CAT_FEATURE_INDICES)
                return self.model.predict_proba(pool)

            self._lime_explainer = LimeTabularExplainer(
                training_data=train_enc[FEATURE_ORDER].values.astype(float),
                feature_names=FEATURE_ORDER,
                categorical_features=CAT_FEATURE_INDICES,
                categorical_names=categorical_names,
                class_names=["No Stroke", "Stroke"],
                discretize_continuous=True,
                random_state=42,
            )
            self._lime_mapping = mapping
            logger.info("LIME explainer initialized.")
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning(f"LIME explainer could not be initialized: {exc}")
            self._lime_explainer = None

    def lime_explain(self, df: pd.DataFrame) -> Dict[str, Any]:
        if not LIME_AVAILABLE:
            return {
                "available": False,
                "message": "LIME is not installed in this environment.",
                "contributions": [],
            }
        if self._lime_explainer is None:
            return {
                "available": False,
                "message": "LIME explanation is not available.",
                "contributions": [],
            }

        try:
            row = df.iloc[0].copy()
            for c in CAT_FEATURE_NAMES:
                if c in self._lime_mapping:
                    row[c] = self._lime_mapping[c][str(row[c])]

            instance = row[FEATURE_ORDER].values.astype(float)

            def predict_fn(data):
                data = pd.DataFrame(data, columns=FEATURE_ORDER)
                data_dec = data.copy()
                for c in CAT_FEATURE_NAMES:
                    if c in self._lime_mapping:
                        inv = {i: v for v, i in self._lime_mapping[c].items()}
                        data_dec[c] = data[c].astype(float).map(inv)
                pool = Pool(data_dec, cat_features=CAT_FEATURE_INDICES)
                return self.model.predict_proba(pool)

            exp = self._lime_explainer.explain_instance(
                instance, predict_fn, num_features=len(FEATURE_ORDER)
            )

            contributions: List[Dict[str, Any]] = []
            for feature_desc, weight in exp.as_list(label=1):
                contributions.append({
                    "feature": feature_desc,
                    "weight": round(float(weight), 6),
                    "direction": "positive" if weight > 0 else ("negative" if weight < 0 else "neutral"),
                })

            return {
                "available": True,
                "message": (
                    "LIME local surrogate-model contributions for this patient. "
                    "Weights reflect the local linear approximation, not the "
                    "global CatBoost model."
                ),
                "contributions": contributions,
                "intercept": round(float(exp.intercept[1]), 6),
            }
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("LIME explanation failed")
            return {
                "available": False,
                "message": f"LIME explanation could not be computed: {exc}",
                "contributions": [],
            }