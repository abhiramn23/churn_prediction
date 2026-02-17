"""
==========================================================
PYDANTIC SCHEMAS — schemas.py
==========================================================

WHAT THIS FILE DOES:
--------------------
Defines the "shape" of data going IN and OUT of our API.

Think of schemas like forms:
- CustomerData = the form a user fills in (input)
- PredictionResponse = the result they get back (output)

WHY WE NEED THIS:
-----------------
- FastAPI uses these schemas to validate input data automatically
- If someone sends wrong data (e.g., text instead of a number),
  FastAPI will return a helpful error message
- They also generate nice API documentation at /docs
==========================================================
"""

from pydantic import BaseModel, Field
from typing import Optional, List


class CustomerData(BaseModel):
    """
    Input schema: data about ONE customer.

    Each field corresponds to a column in our training dataset.
    The 'Field' function adds descriptions and validations.
    """
    gender: str = Field(
        ...,
        description="Customer gender: 'Male' or 'Female'",
        examples=["Male"]
    )
    senior_citizen: int = Field(
        ...,
        description="Is the customer a senior citizen? 0 = No, 1 = Yes",
        ge=0, le=1,
        examples=[0]
    )
    tenure: int = Field(
        ...,
        description="Number of months the customer has been with the company",
        ge=0,
        examples=[12]
    )
    monthly_charges: float = Field(
        ...,
        description="The amount charged to the customer monthly (in dollars)",
        ge=0,
        examples=[29.85]
    )
    total_charges: float = Field(
        ...,
        description="The total amount charged to the customer (in dollars)",
        ge=0,
        examples=[358.20]
    )
    contract_type: str = Field(
        ...,
        description="Type of contract: 'Month-to-month', 'One year', or 'Two year'",
        examples=["Month-to-month"]
    )
    internet_service: str = Field(
        ...,
        description="Internet service type: 'DSL', 'Fiber optic', or 'No'",
        examples=["DSL"]
    )
    payment_method: str = Field(
        ...,
        description="Payment method: 'Electronic check', 'Mailed check', 'Bank transfer', or 'Credit card'",
        examples=["Electronic check"]
    )


class PredictionResponse(BaseModel):
    """
    Output schema: the prediction result for ONE customer.
    """
    prediction: int = Field(
        ...,
        description="0 = customer will NOT churn, 1 = customer WILL churn"
    )
    prediction_label: str = Field(
        ...,
        description="Human-readable label: 'Will Churn' or 'Will Not Churn'"
    )
    confidence: float = Field(
        ...,
        description="How confident the model is in this prediction (0.0 to 1.0)"
    )
    message: str = Field(
        ...,
        description="A friendly message explaining the prediction"
    )


class BatchPredictionRequest(BaseModel):
    """
    Input schema: predict churn for MULTIPLE customers at once.
    """
    customers: List[CustomerData] = Field(
        ...,
        description="A list of customer data objects"
    )


class BatchPredictionResponse(BaseModel):
    """
    Output schema: predictions for multiple customers.
    """
    predictions: List[PredictionResponse] = Field(
        ...,
        description="A list of prediction results"
    )
    total_customers: int = Field(
        ...,
        description="Total number of customers processed"
    )
    churn_count: int = Field(
        ...,
        description="Number of customers predicted to churn"
    )
    churn_rate: float = Field(
        ...,
        description="Percentage of customers predicted to churn"
    )


class HealthResponse(BaseModel):
    """
    Output schema: health check endpoint response.
    """
    status: str = Field(..., description="Service status")
    model_loaded: bool = Field(..., description="Whether the ML model is loaded")
    version: str = Field(..., description="API version")


class ErrorResponse(BaseModel):
    """
    Output schema: error response.
    """
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
