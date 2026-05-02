from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
def health_check():

    return {
        "api": "running",
        "status": "healthy" # 200 OK status is implied by successful response, but we include "status": "healthy" in the JSON body to provide a clear and explicit indication of the API's health status, which can be useful for monitoring and debugging purposes.
    }