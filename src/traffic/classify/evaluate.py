import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold


def crossval_scores(clf, X, y, k=10, repeats=3, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    for r in range(repeats):
        skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=int(rng.integers(1e9)))
        fold_scores = []
        for tr, te in skf.split(X, y):
            clf.fit(X[tr], y[tr])
            p = clf.predict(X[te])
            fold_scores.append(balanced_accuracy_score(y[te], p))
        scores.append(float(np.mean(fold_scores)))
    return np.array(scores, dtype=np.float32)


def mann_whitney_better(a, b, p=0.05):
    return mannwhitneyu(a, b, alternative="greater").pvalue < p
