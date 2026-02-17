"""
==========================================================
ML MODEL LOADER — model.py
==========================================================

WHAT THIS FILE DOES:
--------------------
1. Loads the trained model (model.pkl) from disk when the server starts
2. Loads the feature names and label encoders
3. Provides a predict() function that takes customer data and returns
   a churn prediction

WHY SEPARATE FILE?
------------------
By isolating the ML logic here, you can:
- Swap models without touching the API code
- Test the model independently
- Keep main.py clean and focused on HTTP handling
==========================================================
"""

import joblib
import pandas as pd
import numpy as np
import os
from typing import Dict, Any, Optional


class ChurnModel:
    """
    Wraps the trained ML model and provides prediction methods.

    Usage:
        model = ChurnModel()
        model.load()
        result = model.predict(customer_data_dict)
    """

    def __init__(self):
        """Initialize with empty model — call load() before predicting."""
        self.model = None
        self.feature_names = None
        self.label_encoders = None
        self.is_loaded = False

    def load(self, models_dir: Optional[str] = None) -> None:
        """
        Load the trained model, feature names, and label encoders from disk.

        Args:
            models_dir: Path to the models directory. If None, uses default path.

        Raises:
            FileNotFoundError: If model files are not found.
        """
        # Default: look in the 'models' folder one level up from backend/
        if models_dir is None:
            # This file is at: backend/app/model.py
            # Models are at:   models/
            current_dir = os.path.dirname(os.path.abspath(__file__))
            models_dir = os.path.join(current_dir, "..", "..", "models")

        # Check that the model file exists
        model_path = os.path.join(models_dir, "model.pkl")
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at: {model_path}\n"
                f"Please run 'python model_training/train.py' first to train the model."
            )

        # Load all three files
        print(f"📂 Loading model from: {model_path}")
        self.model = joblib.load(model_path)
        self.feature_names = joblib.load(os.path.join(models_dir, "feature_names.pkl"))
        self.label_encoders = joblib.load(os.path.join(models_dir, "label_encoders.pkl"))
        self.is_loaded = True
        print(f"✅ Model loaded successfully! Features: {self.feature_names}")

    def predict(self, customer_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a churn prediction for a single customer.

        Args:
            customer_data: Dictionary with customer features, e.g.:
                {
                    "gender": "Male",
                    "senior_citizen": 0,
                    "tenure": 12,
                    "monthly_charges": 29.85,
                    "total_charges": 358.20,
                    "contract_type": "Month-to-month",
                    "internet_service": "DSL",
                    "payment_method": "Electronic check"
                }

        Returns:
            Dictionary with prediction, confidence, and message.

        Raises:
            RuntimeError: If model is not loaded yet.
            ValueError: If input data is invalid.
        """
        # Safety check — make sure the model is loaded
        if not self.is_loaded:
            raise RuntimeError("Model is not loaded. Call load() first.")

        try:
            # Step 1: Create a DataFrame from the input dictionary
            # The model expects data in the same format it was trained on
            df = pd.DataFrame([customer_data])

            # Step 2: Encode categorical columns using the SAME encoders from training
            # This ensures "Male" gets the SAME number as during training
            for col, encoder in self.label_encoders.items():
                if col in df.columns:
                    # Handle unseen categories gracefully
                    known_classes = set(encoder.classes_)
                    df[col] = df[col].apply(
                        lambda x: encoder.transform([x])[0] if x in known_classes else -1
                    )

            # Step 3: Reorder columns to match training order
            df = df[self.feature_names]

            # Step 4: Make prediction
            prediction = int(self.model.predict(df)[0])

            # Step 5: Get prediction probability (confidence)
            # predict_proba returns [prob_class_0, prob_class_1]
            probabilities = self.model.predict_proba(df)[0]
            confidence = float(max(probabilities))

            # Step 6: Create human-friendly response
            prediction_label = "Will Churn" if prediction == 1 else "Will Not Churn"

            if prediction == 1:
                message = (
                    f"⚠️ This customer is likely to churn "
                    f"(confidence: {confidence:.1%}). "
                    f"Consider offering retention incentives."
                )
            else:
                message = (
                    f"✅ This customer is likely to stay "
                    f"(confidence: {confidence:.1%}). "
                    f"Keep up the good service!"
                )

            return {
                "prediction": prediction,
                "prediction_label": prediction_label,
                "confidence": round(confidence, 4),
                "message": message
            }

        except KeyError as e:
            raise ValueError(f"Missing required feature: {e}")
        except Exception as e:
            raise ValueError(f"Prediction failed: {str(e)}")


# ──────────────────────────────────────────────────────────
# Create a SINGLETON instance of the model
# This means the model is loaded ONCE when the server starts,
# not every time someone makes a prediction request
# ──────────────────────────────────────────────────────────
churn_model = ChurnModel()
