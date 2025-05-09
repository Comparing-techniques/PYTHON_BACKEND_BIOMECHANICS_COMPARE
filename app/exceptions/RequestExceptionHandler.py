from app.main import app_principal

from fastapi import Request
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from ..handlers import logger


@app_principal.exception_handler(RequestValidationError)
async def request_exception_handler(request: Request, exc: RequestValidationError):
    """
    Custom exception handler for request validation errors.
    """
    # Log the error details here if needed
    # For example: logger.error(f"Validation error: {exc.errors()}")

    errors = []

    for e in exc.errors():
        loc = e.get("loc", [])
        msg = e.get("msg")
        errors.append({"field": loc[-1] if loc else None, "error": msg})
    # Return a custom response
    return JSONResponse(
            status_code=422,
            content={"message": "Bad files and parameters", "errors": errors}
        )
    
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    errors = [{"field": e["loc"][-1], "msg": e["msg"]} for e in exc.errors()]
    logger.error(f"Validation errors: {errors}")
    return JSONResponse(status_code=422, content={"message": "Invalid request", "errors": errors})