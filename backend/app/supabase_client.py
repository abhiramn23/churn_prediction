"""
==========================================================
SUPABASE CLIENT — supabase_client.py
==========================================================

WHAT THIS FILE DOES:
--------------------
1. Connects to your Supabase project (cloud database + auth)
2. Provides helper functions to store/retrieve prediction data
3. Handles Supabase errors gracefully

WHAT IS SUPABASE?
-----------------
Supabase is an open-source alternative to Firebase. It gives you:
- A PostgreSQL database (to store data)
- Authentication (user signup/login)
- REST API (auto-generated from your database)
- All for free on the free tier!

SETUP REQUIRED:
---------------
1. Go to https://supabase.com and create a free account
2. Create a new project
3. Go to Settings → API → Copy your:
   - Project URL  (e.g., https://abcdefg.supabase.co)
   - anon/public key (a long string starting with 'eyJ...')
4. Create a .env file in the backend/ folder with:
   SUPABASE_URL=https://your-project.supabase.co
   SUPABASE_KEY=your-anon-key

DATABASE TABLE SETUP:
---------------------
In the Supabase dashboard, go to SQL Editor and run:

CREATE TABLE predictions (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id),
    customer_data JSONB NOT NULL,
    prediction INTEGER NOT NULL,
    prediction_label TEXT NOT NULL,
    confidence FLOAT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Enable Row Level Security (RLS)
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;

-- Allow users to see only their own predictions
CREATE POLICY "Users can view own predictions"
    ON predictions FOR SELECT
    USING (auth.uid() = user_id);

-- Allow users to insert their own predictions
CREATE POLICY "Users can insert own predictions"
    ON predictions FOR INSERT
    WITH CHECK (auth.uid() = user_id);
==========================================================
"""

import os
from typing import Dict, Any, Optional, List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def get_supabase_client():
    """
    Create and return a Supabase client.

    Returns:
        Supabase client instance, or None if credentials are not set.
    """
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")

    if not supabase_url or not supabase_key:
        print("⚠️  WARNING: SUPABASE_URL or SUPABASE_KEY not set in .env file.")
        print("   The app will work without Supabase, but predictions won't be saved.")
        print("   See supabase_client.py for setup instructions.")
        return None

    try:
        from supabase import create_client, Client
        client: Client = create_client(supabase_url, supabase_key)
        print("✅ Connected to Supabase!")
        return client
    except ImportError:
        print("⚠️  WARNING: 'supabase' package not installed. Run: pip install supabase")
        return None
    except Exception as e:
        print(f"❌ Failed to connect to Supabase: {e}")
        return None


# Create a singleton client
supabase = get_supabase_client()


def store_prediction(
    user_id: str,
    customer_data: Dict[str, Any],
    prediction: int,
    prediction_label: str,
    confidence: float
) -> Optional[Dict]:
    """
    Store a prediction result in the Supabase 'predictions' table.

    Args:
        user_id: The UUID of the authenticated user
        customer_data: The input customer data (stored as JSON)
        prediction: 0 or 1
        prediction_label: "Will Churn" or "Will Not Churn"
        confidence: Model confidence score

    Returns:
        The inserted row data, or None if Supabase is not configured.
    """
    if supabase is None:
        print("⚠️  Supabase not configured — skipping prediction storage.")
        return None

    try:
        data = {
            "user_id": user_id,
            "customer_data": customer_data,
            "prediction": prediction,
            "prediction_label": prediction_label,
            "confidence": confidence
        }

        result = supabase.table("predictions").insert(data).execute()
        print(f"✅ Prediction stored in Supabase (user: {user_id[:8]}...)")
        return result.data
    except Exception as e:
        print(f"❌ Failed to store prediction: {e}")
        return None


def get_user_predictions(user_id: str, limit: int = 50) -> List[Dict]:
    """
    Retrieve prediction history for a specific user.

    Args:
        user_id: The UUID of the authenticated user
        limit: Maximum number of predictions to return

    Returns:
        List of prediction records, or empty list if Supabase is not configured.
    """
    if supabase is None:
        return []

    try:
        result = (
            supabase.table("predictions")
            .select("*")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return result.data
    except Exception as e:
        print(f"❌ Failed to fetch predictions: {e}")
        return []


def verify_user_token(access_token: str) -> Optional[Dict]:
    """
    Verify a Supabase JWT token and return the user info.

    Args:
        access_token: The JWT token from the Authorization header

    Returns:
        User info dictionary, or None if token is invalid.
    """
    if supabase is None:
        return None

    try:
        user_response = supabase.auth.get_user(access_token)
        if user_response and user_response.user:
            return {
                "id": str(user_response.user.id),
                "email": user_response.user.email,
            }
        return None
    except Exception as e:
        print(f"❌ Token verification failed: {e}")
        return None
