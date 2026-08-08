import os
from typing import Optional

#reads the environment variable for the API key
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader

API_KEY = os.environ.get("API_KEY")

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
#expect client to send the API key in the header named "X-API-Key"

def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key",
        )