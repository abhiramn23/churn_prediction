"""
==========================================================
AUTHENTICATION MIDDLEWARE — auth.py
==========================================================

WHAT THIS FILE DOES:
--------------------
Provides a FastAPI "dependency" that checks if a user is logged in.

HOW IT WORKS:
-------------
1. The frontend sends the user's Supabase token in the HTTP header:
   Authorization: Bearer <token>
2. This middleware extracts that token
3. Verifies it with Supabase
4. Returns the user info (id, email) to the endpoint

WHY A SEPARATE FILE?
--------------------
- Keeps auth logic centralized
- Can be reused across multiple endpoints
- Easy to modify auth strategy later
==========================================================
"""

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict
from app.supabase_client import verify_user_token

# HTTPBearer extracts the token from the "Authorization: Bearer <token>" header
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
) -> Optional[Dict]:
    """
    FastAPI dependency that extracts and verifies the user's auth token.

    Usage in an endpoint:
        @app.get("/protected")
        async def protected_route(user = Depends(get_current_user)):
            if user is None:
                raise HTTPException(status_code=401, detail="Not authenticated")
            return {"email": user["email"]}

    Returns:
        User dict with 'id' and 'email', or None if not authenticated.
    """
    # If no credentials provided, return None (unauthenticated)
    if credentials is None:
        return None

    # Extract the token string
    token = credentials.credentials

    # Verify with Supabase
    user = verify_user_token(token)
    return user


async def require_auth(
    user: Optional[Dict] = Depends(get_current_user)
) -> Dict:
    """
    Stricter version — REQUIRES the user to be authenticated.
    Raises 401 error if not logged in.

    Usage:
        @app.get("/must-be-logged-in")
        async def protected_route(user = Depends(require_auth)):
            return {"email": user["email"]}
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required. Please provide a valid Supabase token.",
            headers={"WWW-Authenticate": "Bearer"}
        )
    return user
