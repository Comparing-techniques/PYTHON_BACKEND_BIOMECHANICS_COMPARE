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
    return await execute_comparison(request.base_excel_file, request.excel_file_compare, request.joint_id)