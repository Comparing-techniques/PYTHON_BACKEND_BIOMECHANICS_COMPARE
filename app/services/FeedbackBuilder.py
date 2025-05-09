
def build_feedback(joint_errors: dict, clusters: dict, thresholds: dict):
    """
    Genera mensaje de retroalimentación basado en errores y umbrales.
    """
    feedback = {}
    for joint, errors in joint_errors.items():
        rmse = errors.get('rmse')
        if rmse > thresholds.get(joint, float('inf')):
            feedback[joint] = f"El movimiento en {joint} excede el rango esperado (RMSE={rmse:.2f})."
        else:
            feedback[joint] = f"{joint} dentro de límites normales."
    return feedback