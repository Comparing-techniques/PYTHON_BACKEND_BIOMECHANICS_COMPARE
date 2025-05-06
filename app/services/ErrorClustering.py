from sklearn.cluster import KMeans
import numpy as np

class ErrorClusteringService:
    @staticmethod
    def cluster_errors(error_sequence: np.ndarray, n_clusters: int = 3):
        """
        Agrupa errores por similitud temporal.
        """
        km = KMeans(n_clusters=n_clusters)
        labels = km.fit_predict(error_sequence.reshape(-1, 1))
        return labels