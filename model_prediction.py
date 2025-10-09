import json
import joblib
import numpy as np
from typing import Dict, List, Sequence, Tuple


_DEFAULT_MODEL_PATH = "models/calibrated_clf.joblib"
_DEFAULT_CRITERIA_PATH = "default_criteria.json"

# Prior from analysis.ipynb
_PRIOR_DISTRIBUTION: Dict[str, float] = {
    "hungry": 0.40,
    "tired": 0.25,
    "discomfort": 0.15,
    "burping": 0.12,
    "belly_pain": 0.08,
}

# Class label order used during training (LabelEncoder sorts alphabetically)
_ALPHABETICAL_CLASS_LABELS: List[str] = [
    "belly_pain",
    "burping",
    "discomfort",
    "hungry",
    "tired",
]


def _load_criteria(criteria_path: str = _DEFAULT_CRITERIA_PATH) -> List[str]:
    with open(criteria_path, "r") as f:
        data = json.load(f)
    return list(data.get("criteria", []))


def _scores_to_feature_vector(
    scores: Sequence[Tuple[str, float]],
    ordered_criteria: Sequence[str],
) -> np.ndarray:
    score_map: Dict[str, float] = {criterion: float(value) for criterion, value in scores}
    return np.array([score_map.get(c, 0.0) for c in ordered_criteria], dtype=np.float32)


def _get_class_labels(model) -> List[str]:
    classes = getattr(model, "classes_", None)
    if classes is None:
        return list(_ALPHABETICAL_CLASS_LABELS)

    # If trained with LabelEncoder, classes_ are integers 0..n-1 in alphabetical label order
    if np.issubdtype(np.array(classes).dtype, np.integer):
        return list(_ALPHABETICAL_CLASS_LABELS)

    # Otherwise assume classes_ contains the string labels
    return [str(c) for c in classes]


def _vector_from_prior(labels: Sequence[str], prior: Dict[str, float]) -> np.ndarray:
    values: List[float] = []
    for label in labels:
        if label not in prior:
            raise ValueError(f"Missing prior for class '{label}'")
        values.append(float(prior[label]))
    vec = np.array(values, dtype=np.float64)
    total = float(vec.sum())
    return vec / total if total > 0 else vec


def predict_distribution_from_rater_scores(
    rater_scores: Sequence[Tuple[str, float]],
    model_path: str = _DEFAULT_MODEL_PATH,
    criteria_path: str = _DEFAULT_CRITERIA_PATH,
    prior: Dict[str, float] | None = None,
) -> Dict[str, float]:
    """
    Compute class probability distribution by combining calibrated model predictions
    with the prior from analysis.ipynb via a simple average.

    Args:
        rater_scores: Sequence of (criterion, score) pairs from rater.compute_similarity.
        model_path: Filesystem path to the saved calibrated classifier.
        criteria_path: Path to criteria JSON used to define feature order.
        prior: Optional prior distribution mapping class -> probability. If None, uses _PRIOR_DISTRIBUTION.

    Returns:
        Dict mapping class name -> averaged probability across the five classes.
    """
    criteria = _load_criteria(criteria_path)
    features = _scores_to_feature_vector(rater_scores, criteria)

    model = joblib.load(model_path)
    labels = _get_class_labels(model)

    X = features.reshape(1, -1)
    pred_proba: np.ndarray = model.predict_proba(X)[0]

    # Align prior to label order
    prior_dict = prior or _PRIOR_DISTRIBUTION
    prior_vec = _vector_from_prior(labels, prior_dict)

    averaged = 0.8 * pred_proba.astype(np.float64) + 0.2 * prior_vec
    # Numerical stability: renormalize
    total = float(averaged.sum())
    if total > 0:
        averaged = averaged / total

    return {label: float(prob) for label, prob in zip(labels, averaged)}


def predict_top_label_from_rater_scores(
    rater_scores: Sequence[Tuple[str, float]],
    model_path: str = _DEFAULT_MODEL_PATH,
    criteria_path: str = _DEFAULT_CRITERIA_PATH,
    prior: Dict[str, float] | None = None,
) -> Tuple[str, float, Dict[str, float]]:
    """
    Convenience helper returning the top class, its probability, and the full distribution.
    """
    dist = predict_distribution_from_rater_scores(
        rater_scores=rater_scores,
        model_path=model_path,
        criteria_path=criteria_path,
        prior=prior,
    )
    labels = list(dist.keys())
    probs = np.array([dist[k] for k in labels], dtype=np.float64)
    idx = int(np.argmax(probs))
    return labels[idx], float(probs[idx]), dist


