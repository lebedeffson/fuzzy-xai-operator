from __future__ import annotations


def proxy_action_label(
    predicted_risk: float,
    uncertainty: float,
    i_pre: float,
    rupture: bool,
    critical_rupture: bool,
) -> str:
    """Heuristic proxy label for observer action agreement.

    This is an evaluation proxy rule, not expert clinical ground truth.
    """
    if critical_rupture:
        return 'block'
    if rupture and predicted_risk >= 0.80:
        return 'defer_to_human'
    if predicted_risk >= 0.75:
        return 'defer_to_human'
    if uncertainty >= 0.45 or i_pre < 0.65:
        return 'request_more_data'
    if predicted_risk >= 0.35:
        return 'lower_confidence'
    return 'accept'


def reference_action_label(
    predicted_risk: float,
    uncertainty: float,
    i_pre: float,
    rupture: bool,
    critical_rupture: bool,
    *,
    y_true: int | None = None,
    y_pred: int | None = None,
) -> str:
    """Independent evaluation target for observer actions.

    Unlike proxy_action_label, this target can include prediction correctness
    signal on labeled datasets and is used for fairer benchmark comparisons.
    """
    if critical_rupture:
        return 'block'
    if (y_true is not None) and (y_pred is not None) and (int(y_true) != int(y_pred)) and predicted_risk >= 0.65:
        return 'defer_to_human'
    if rupture or uncertainty >= 0.45 or i_pre < 0.65:
        return 'request_more_data'
    if predicted_risk >= 0.75:
        return 'defer_to_human'
    if predicted_risk >= 0.35:
        return 'lower_confidence'
    return 'accept'
