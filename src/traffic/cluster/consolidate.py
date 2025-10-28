import numpy as np
from sklearn.cluster import KMeans


def consolidate_by_exit(exit_points: np.ndarray, k: int = 8):
    """Second-stage consolidation: K-Means on exit representatives."""
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    exit_labels = km.fit_predict(exit_points)
    return exit_labels, km
