"""
Pydantic schemas for the StrokeGuard API.

Defines the request/response contracts for the prediction endpoint.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, field_validator, model_validator


class PatientInput(BaseModel):
    """
    Validated patient assessment input.

    Field names, types and categorical values match the trained
    CatBoost model's feature schema exactly. No extra inputs are
    invented.
    """

    gender: str = Field(
        ...,
        description="Patient gender.",
        examples=["Male", "Female", "Other"],
    )
    age: float = Field(
        ...,
        ge=1,
        le=120,
        description="Patient age in years.",
        examples=[55],
    )
    hypertension: int = Field(
        ...,
        ge=0,
        le=1,
        description="Whether the patient has hypertension (0 = No, 1 = Yes).",
        examples=[0, 1],
    )
    heart_disease: int = Field(
        ...,
        ge=0,
        le=1,
        description="Whether the patient has heart disease (0 = No, 1 = Yes).",
        examples=[0, 1],
    )
    ever_married: str = Field(
        ...,
        description="Whether the patient has ever been married.",
        examples=["Yes", "No"],
    )
    work_type: str = Field(
        ...,
        description="Type of work the patient does.",
        examples=["Private", "Self-employed", "Govt_job", "Children", "Never_worked"],
    )
    residence_type: str = Field(
        ...,
        description="Urban or rural residence.",
        examples=["Urban", "Rural"],
    )
    avg_glucose_level: float = Field(
        ...,
        ge=0,
        le=500,
        description="Average blood glucose level (mg/dL).",
        examples=[110.0],
    )
    bmi: Optional[float] = Field(
        None,
        ge=10,
        le=70,
        description="Body mass index (kg/m^2). Leave empty if unknown.",
        examples=[27.0, None],
    )
    smoking_status: str = Field(
        ...,
        description="Patient smoking status.",
        examples=["formerly smoked", "never smoked", "smokes", "Unknown"],
    )

    @field_validator("gender")
    @classmethod
    def _validate_gender(cls, v: str) -> str:
        allowed = {"Male", "Female", "Other"}
        if v not in allowed:
            raise ValueError(f"gender must be one of {sorted(allowed)}")
        return v

    @field_validator("ever_married")
    @classmethod
    def _validate_ever_married(cls, v: str) -> str:
        allowed = {"Yes", "No"}
        if v not in allowed:
            raise ValueError(f"ever_married must be one of {sorted(allowed)}")
        return v

    @field_validator("work_type")
    @classmethod
    def _validate_work_type(cls, v: str) -> str:
        allowed = {"Private", "Self-employed", "Govt_job", "Children", "Never_worked"}
        if v not in allowed:
            raise ValueError(f"work_type must be one of {sorted(allowed)}")
        return v

    @field_validator("residence_type")
    @classmethod
    def _validate_residence_type(cls, v: str) -> str:
        allowed = {"Urban", "Rural"}
        if v not in allowed:
            raise ValueError(f"residence_type must be one of {sorted(allowed)}")
        return v

    @field_validator("smoking_status")
    @classmethod
    def _validate_smoking_status(cls, v: str) -> str:
        allowed = {"formerly smoked", "never smoked", "smokes", "Unknown"}
        if v not in allowed:
            raise ValueError(f"smoking_status must be one of {sorted(allowed)}")
        return v

    @model_validator(mode="after")
    def _derive_bmi_missing(self) -> "PatientInput":
        """
        Derive the bmi_missing indicator. If BMI is not supplied,
        bmi_missing is set to 1 and bmi is left as NaN for the model
        (CatBoost handles missing values natively). Zero is never
        silently substituted for a missing BMI.
        """
        self.bmi_missing = 0 if self.bmi is not None else 1
        return self

    # Populated by the validator above
    bmi_missing: int = 0


class ShapContribution(BaseModel):
    feature: str
    value: float
    direction: str  # "positive" | "negative"


class LimeContribution(BaseModel):
    feature: str
    weight: float
    direction: str  # "positive" | "negative"


class ExplanationBlock(BaseModel):
    available: bool
    message: Optional[str] = None
    contributions: Optional[List[Dict[str, Any]]] = None
    base_value: Optional[float] = None


class PredictionResponse(BaseModel):
    prediction: int  # 0 = No Stroke, 1 = Stroke
    prediction_label: str
    probability: float
    model_name: str
    timestamp: str
    shap_explanation: ExplanationBlock
    lime_explanation: ExplanationBlock
    patient_summary: Dict[str, Any]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    model_name: str