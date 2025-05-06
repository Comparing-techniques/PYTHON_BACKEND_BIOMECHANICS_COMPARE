import numpy as np
from ..utils.preprocessing_helpers import create_dict_with_data, convertir_a_dict_articulaciones

class JointCentersAnglesService:
    @staticmethod
    def extract_joint_data(marker_dict: dict, step: int = 1):
        """
        Genera diccionario de articulaciones para un sujeto.
        """
        joints_calc = create_dict_with_data(marker_dict)
        articulaciones = convertir_a_dict_articulaciones(marker_dict, joints_calc, step=step)
        return articulaciones