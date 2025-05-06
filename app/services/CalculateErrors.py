import numpy as np

class CalculateErrorsService:
    @staticmethod
    def rmse(a: np.ndarray, b: np.ndarray):
        """
        Calcula RMSE entre dos secuencias.
        """
        return np.sqrt(np.mean((a - b) ** 2))

    @staticmethod
    def error_metrics(aligned_a: np.ndarray, aligned_b: np.ndarray):
        """
        Devuelve diccionario con RMSE y otras métricas.
        """
        return {
            'rmse': CalculateErrorsService.rmse(aligned_a, aligned_b),
            'mean_diff': np.mean(aligned_a - aligned_b)
        }