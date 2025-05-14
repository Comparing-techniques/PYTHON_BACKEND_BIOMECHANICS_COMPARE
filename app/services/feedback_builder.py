from ..handlers.logger import logger
from ..utils.errors_feedback_builder import (
    FEEDBACK_JOINT_EXCEEDS_THRESHOLD,
    FEEDBACK_JOINT_WITHIN_LIMITS
)


"""
Genera mensaje de retroalimentación basado en errores y umbrales.
"""

def build_feedback(joint_errors: dict, clusters: dict, thresholds: dict):
    feedback = {}
    for joint, errors in joint_errors.items():
        rmse = errors.get('rmse')
        if rmse > thresholds.get(joint, float('inf')):
            feedback[joint] = FEEDBACK_JOINT_EXCEEDS_THRESHOLD.format(joint, rmse)
        else:
            feedback[joint] = FEEDBACK_JOINT_WITHIN_LIMITS.format(joint)
    return feedback