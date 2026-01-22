"""
Authentication middleware for LEGO RAG Backend.

Provides JWT-based authentication and role-based access control.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import Security, HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import jwt, JWTError
from passlib.context import CryptContext
import os
from loguru import logger

# Security configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "CHANGE_THIS_IN_PRODUCTION_VERY_IMPORTANT")
if SECRET_KEY == "CHANGE_THIS_IN_PRODUCTION_VERY_IMPORTANT":
    logger.warning("⚠️  Using default JWT_SECRET_KEY! Set JWT_SECRET_KEY environment variable in production!")

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

# Password hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# HTTP Bearer token authentication
security = HTTPBearer()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hashed password."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generate password hash."""
    return pwd_context.hash(password)


def create_access_token(data: Dict, expires_delta: Optional[timedelta] = None) -> str:
    """
    Create a JWT access token.

    Args:
        data: Dictionary with user data (must include 'sub' for subject/user_id)
        expires_delta: Token expiration time (defaults to 24 hours)

    Returns:
        Encoded JWT token
    """
    to_encode = data.copy()

    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)) -> Dict:
    """
    Verify JWT token and return payload.

    Args:
        credentials: HTTP Bearer token from Authorization header

    Returns:
        Token payload dictionary

    Raises:
        HTTPException: If token is invalid or expired
    """
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: str = payload.get("sub")

        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        return payload

    except JWTError as e:
        logger.warning(f"JWT validation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def require_admin(token: Dict = Depends(verify_token)) -> Dict:
    """
    Require admin role for endpoint access.

    Args:
        token: Verified JWT token payload

    Returns:
        Token payload if user is admin

    Raises:
        HTTPException: If user doesn't have admin role
    """
    role = token.get("role")
    if role != "admin":
        logger.warning(f"Access denied: User {token.get('sub')} attempted admin action without privileges")
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    return token


async def get_current_user(token: Dict = Depends(verify_token)) -> Dict:
    """
    Get current authenticated user from token.

    Args:
        token: Verified JWT token payload

    Returns:
        User information from token
    """
    return {
        "user_id": token.get("sub"),
        "role": token.get("role", "user"),
        "email": token.get("email")
    }


# Optional: Simple API key authentication for programmatic access
def verify_api_key(api_key: str) -> bool:
    """
    Verify API key for programmatic access.

    Args:
        api_key: API key to verify

    Returns:
        True if valid, False otherwise
    """
    # In production, store API keys in database with hashing
    valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")
    return api_key in valid_api_keys


async def verify_api_key_or_token(
    credentials: Optional[HTTPAuthorizationCredentials] = Security(security)
) -> Dict:
    """
    Accept either JWT token or API key for authentication.

    Args:
        credentials: HTTP Bearer token (JWT or API key)

    Returns:
        User/API information

    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required"
        )

    token = credentials.credentials

    # Try JWT first
    try:
        return await verify_token(credentials)
    except HTTPException:
        pass

    # Try API key
    if verify_api_key(token):
        return {
            "sub": "api_key_user",
            "role": "user",
            "auth_method": "api_key"
        }

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials"
    )
