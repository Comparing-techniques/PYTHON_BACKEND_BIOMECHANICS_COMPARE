import numpy as np

class AccelerationSpeedService:
    @staticmethod
    def compute_velocity(position: np.ndarray, time_ms: np.ndarray):
        """
        Calcula velocidades (m/s) a partir de posiciones (m) y tiempos (ms).
        """
        dt = np.diff(time_ms) / 1000.0
        vel = np.diff(position, axis=0) / dt[:, None]
        return vel

    @staticmethod
    def compute_acceleration(velocity: np.ndarray, time_ms: np.ndarray):
        """
        Calcula aceleraciones (m/s²) a partir de velocidades y tiempos.
        """
        dt = np.diff(time_ms[:-1]) / 1000.0
        acc = np.diff(velocity, axis=0) / dt[:, None]
        return acc