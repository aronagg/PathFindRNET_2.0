from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def make_model(kind: str):
    k = kind.lower()
    if k == "knn":
        return KNeighborsClassifier(n_neighbors=7, weights="distance")
    if k == "svm":
        return SVC(kernel="rbf", probability=True, C=2.0, gamma="scale")
    if k == "dt":
        return DecisionTreeClassifier(max_depth=14, random_state=42)
    if k == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(64, 64),
            activation="relu",
            alpha=1e-4,
            max_iter=500,
            early_stopping=True,
            random_state=42,
        )
    raise ValueError(f"Unknown classifier kind: {kind}")
