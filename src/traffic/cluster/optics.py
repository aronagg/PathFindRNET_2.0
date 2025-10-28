import numpy as np
from sklearn.cluster import OPTICS


def optics_cluster(entry_exit_points: np.ndarray, min_samples: int = 30, xi: float = 0.05):
    """Run OPTICS on 4D [x_entry, y_entry, x_exit, y_exit] points."""
    model = OPTICS(min_samples=min_samples, xi=xi)
    labels = model.fit_predict(entry_exit_points)
    return labels, model
