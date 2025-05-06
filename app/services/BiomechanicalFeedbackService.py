import numpy as np
from ..services import (
    ProcessingFilesService, 
    JointCentersAnglesService, 
    AlignMovementsService, 
    CalculateErrorsService, 
    FeedbackBuilderService, 
    ErrorClusteringService
)


class BiomechanicsFeedbackService:
    @staticmethod
    async def compare(base_file, compare_file, joint_id: int):
        # 1. Parsear datos de marcadores
        base_markers = await ProcessingFilesService.parse_marker_data(base_file)
        cmp_markers  = await ProcessingFilesService.parse_marker_data(compare_file)

        # 2. Extraer datos de articulaciones
        base_joints = JointCentersAnglesService.extract_joint_data(base_markers)
        cmp_joints  = JointCentersAnglesService.extract_joint_data(cmp_markers)

        joint_key = list(base_joints['data'].keys())[joint_id - 1]

        # 3. Alinear secuencias de rotación para la articulación
        seg_a = np.stack([base_joints['data'][joint_key][k] for k in ['xrot','yrot','zrot','wrot']], axis=1)
        seg_b = np.stack([cmp_joints ['data'][joint_key][k] for k in ['xrot','yrot','zrot','wrot']], axis=1)
        aligned_a, aligned_b, dist = AlignMovementsService.align_segments(seg_a, seg_b)

        # 4. Calcular errores
        errors = CalculateErrorsService.error_metrics(aligned_a, aligned_b)

        # 5. Agrupar errores temporales
        clusters = ErrorClusteringService.cluster_errors(aligned_a[:,0] - aligned_b[:,0])

        # 6. Construir feedback
        thresholds = {}  # los podrías definir usando rangos_movimientos_con_signos
        feedback = FeedbackBuilderService.build_feedback({joint_key: errors}, {joint_key: clusters}, thresholds)

        return {
            'joint': joint_key,
            'distance': dist,
            'errors': errors,
            'clusters': clusters.tolist(),
            'feedback': feedback[joint_key]
        }