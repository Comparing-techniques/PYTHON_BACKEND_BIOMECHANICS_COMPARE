from app.services.biomechanical_feedback_service import execute_comparison
from ..models.request_model import RequestModel
from ..models import request_model
# from ..services import BiomechanicsFeedbackService


async def biomechanics_feedback_controller(request: request_model):
    """
    Controller function to handle the biomechanics feedback request.
    """
    # Here you would typically process the request and return a response
    # For example, you might save the files or perform some analysis
    # For now, let's just return a simple message
    base_name = request.base_excel_file.filename
    cmp_name  = request.excel_file_compare.filename
    
    await execute_comparison(request.base_excel_file, request.excel_file_compare, request.joint_id)
    return {
        "message": "Biomechanics feedback request received",
        "base_excel_file": base_name,
        "excel_file_compare": cmp_name,
        "joint_id": request.joint_id
    }