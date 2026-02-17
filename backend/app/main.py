"""
==========================================================
FASTAPI MAIN APPLICATION — main.py
==========================================================

WHAT THIS FILE DOES:
--------------------
This is the ENTRY POINT of our backend API. It:

1. Creates the FastAPI application
2. Configures CORS (so the frontend can talk to the backend)
3. Loads the ML model when the server starts
4. Defines API endpoints:
   - GET  /           → Health check (is the server running?)
   - GET  /health     → Detailed health info
   - POST /predict    → Predict churn for ONE customer
   - POST /batch-predict → Predict churn for MULTIPLE customers
   - GET  /predictions → Get prediction history (requires auth)

WHAT IS FASTAPI?
----------------
FastAPI is a modern Python web framework for building APIs.
- It's FAST (hence the name)
- It auto-generates documentation at /docs
- It validates input data using Pydantic schemas
- It supports async/await for high performance

HOW TO RUN:
-----------
    cd backend
    pip install -r requirements.txt
    uvicorn app.main:app --reload

Then open http://localhost:8000/docs to see the interactive API docs.
==========================================================
"""

from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Dict

# Import our custom modules
from app.schemas import (
    CustomerData,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    ErrorResponse
)
from app.model import churn_model
from app.auth import get_current_user, require_auth
from app.supabase_client import store_prediction, get_user_predictions


# ──────────────────────────────────────────────────────────
# STEP 1: Create the FastAPI application
# ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Customer Churn Prediction API",
    description=(
        "🔮 Predict whether a customer will churn (leave) based on their data.\n\n"
        "Built with FastAPI + Scikit-learn. "
        "Part of the Customer Churn Prediction System."
    ),
    version="1.0.0",
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc",     # ReDoc at /redoc
)


# ──────────────────────────────────────────────────────────
# STEP 2: Configure CORS
# ──────────────────────────────────────────────────────────
# CORS = Cross-Origin Resource Sharing
# This allows the Streamlit frontend (running on a DIFFERENT port/domain)
# to make requests to this backend API.
# Without CORS, the browser would BLOCK the frontend from calling the API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # Allow ALL origins (for development)
    allow_credentials=True,
    allow_methods=["*"],       # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],       # Allow all headers (including Authorization)
)


# ──────────────────────────────────────────────────────────
# STEP 3: Load the ML model when the server starts
# ──────────────────────────────────────────────────────────
@app.on_event("startup")
async def load_model():
    """
    This function runs ONCE when the server starts.
    It loads the trained ML model into memory so it's ready for predictions.
    """
    try:
        churn_model.load()
        print("✅ Model loaded successfully on startup!")
    except FileNotFoundError as e:
        print(f"⚠️  WARNING: {e}")
        print("   The API will start but /predict will not work.")
        print("   Run 'python model_training/train.py' to train the model first.")
    except Exception as e:
        print(f"❌ ERROR loading model: {e}")


# ──────────────────────────────────────────────────────────
# ENDPOINT 1: Root — Simple health check
# ──────────────────────────────────────────────────────────
@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint. Returns a welcome message.
    Use this to quickly check if the server is running.
    """
    return {
        "message": "🔮 Customer Churn Prediction API is running!",
        "docs": "Visit /docs for interactive API documentation",
        "health": "/health for detailed status"
    }


# ──────────────────────────────────────────────────────────
# ENDPOINT 2: Health — Detailed health check
# ──────────────────────────────────────────────────────────
@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["Health"]
)
async def health_check():
    """
    Detailed health check. Shows if the model is loaded.
    Used by monitoring tools and deployment platforms.
    """
    return HealthResponse(
        status="healthy",
        model_loaded=churn_model.is_loaded,
        version="1.0.0"
    )


# ──────────────────────────────────────────────────────────
# ENDPOINT 3: Predict — Single customer prediction
# ──────────────────────────────────────────────────────────
@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid input data"},
        500: {"model": ErrorResponse, "description": "Model not loaded"},
    },
    tags=["Predictions"]
)
async def predict_churn(
    customer: CustomerData,
    user: Optional[Dict] = Depends(get_current_user)
):
    """
    🔮 Predict whether a single customer will churn.

    Send customer data and get back:
    - prediction (0 or 1)
    - prediction_label ("Will Churn" or "Will Not Churn")
    - confidence (0.0 to 1.0)
    - message (human-friendly explanation)

    **Authentication is optional** — if you provide a Bearer token,
    the prediction will be saved to your history in Supabase.
    """
    # Check if the model is loaded
    if not churn_model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model is not loaded. Please contact the admin."
        )

    try:
        # Convert Pydantic model to dictionary
        customer_dict = customer.model_dump()

        # Make prediction using our ML model
        result = churn_model.predict(customer_dict)

        # If the user is authenticated, store the prediction in Supabase
        if user is not None:
            store_prediction(
                user_id=user["id"],
                customer_data=customer_dict,
                prediction=result["prediction"],
                prediction_label=result["prediction_label"],
                confidence=result["confidence"]
            )

        return PredictionResponse(**result)

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


# ──────────────────────────────────────────────────────────
# ENDPOINT 4: Batch Predict — Multiple customers at once
# ──────────────────────────────────────────────────────────
@app.post(
    "/batch-predict",
    response_model=BatchPredictionResponse,
    responses={
        400: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
    tags=["Predictions"]
)
async def batch_predict_churn(
    request: BatchPredictionRequest,
    user: Optional[Dict] = Depends(get_current_user)
):
    """
    🔮 Predict churn for MULTIPLE customers at once.

    Send a list of customers and get predictions for all of them.
    Also returns summary statistics (total, churn count, churn rate).
    """
    if not churn_model.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model is not loaded. Please contact the admin."
        )

    try:
        predictions = []
        churn_count = 0

        for customer in request.customers:
            customer_dict = customer.model_dump()
            result = churn_model.predict(customer_dict)
            predictions.append(PredictionResponse(**result))

            if result["prediction"] == 1:
                churn_count += 1

            # Store each prediction if user is authenticated
            if user is not None:
                store_prediction(
                    user_id=user["id"],
                    customer_data=customer_dict,
                    prediction=result["prediction"],
                    prediction_label=result["prediction_label"],
                    confidence=result["confidence"]
                )

        total = len(predictions)
        churn_rate = (churn_count / total * 100) if total > 0 else 0

        return BatchPredictionResponse(
            predictions=predictions,
            total_customers=total,
            churn_count=churn_count,
            churn_rate=round(churn_rate, 2)
        )

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )


# ──────────────────────────────────────────────────────────
# ENDPOINT 5: Prediction History (requires authentication)
# ──────────────────────────────────────────────────────────
@app.get(
    "/predictions",
    tags=["Predictions"]
)
async def prediction_history(
    limit: int = 50,
    user: Dict = Depends(require_auth)
):
    """
    📜 Get your prediction history.

    **Requires authentication** (Bearer token).
    Returns the last N predictions you've made, newest first.
    """
    predictions = get_user_predictions(user["id"], limit=limit)
    return {
        "user_email": user["email"],
        "total": len(predictions),
        "predictions": predictions
    }


# ──────────────────────────────────────────────────────────
# This block runs when you execute: python -m app.main
# (normally you use `uvicorn app.main:app --reload` instead)
# ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
