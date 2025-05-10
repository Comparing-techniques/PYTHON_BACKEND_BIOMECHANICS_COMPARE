from fastapi import APIRouter, Depends
from ..handlers.Logger import logger
from ..models.request_model import RequestModel
from ..controllers.biomechanics_feedback import biomechanics_feedback_controller


router = APIRouter(prefix="/feedback", tags=["BiomechanicsFeedback"])

@router.post("/", response_model=dict)
async def biomechanics_feedback(
    request: RequestModel = Depends(RequestModel.validate)
):
    """
    Endpoint to handle the biomechanics feedback request.
    """
    # Call the controller function to process the request
    return await biomechanics_feedback_controller(request)


__all__ = ["router"]