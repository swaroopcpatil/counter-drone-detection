import numpy as np

from counter_drone.classifier_fit import CUES, auc, fit, score


def test_auc_matches_definition():
    y = np.array([1, 1, 0, 0])
    assert auc(y, np.array([0.9, 0.6, 0.7, 0.1])) == 0.75   # 3 of 4 pos/neg pairs ranked correctly
    assert auc(y, np.array([0.9, 0.8, 0.2, 0.1])) == 1.0


def test_fit_learns_to_trust_an_informative_cue():
    """If 'label' is informative but the hand-set 'flap' cue is misleading (points the wrong
    way), fitting must move 'label' up and 'flap' down relative to the hand-set model."""
    rng = np.random.default_rng(0)
    rows = []
    for _ in range(400):
        y = int(rng.random() < 0.5)
        x = np.zeros(len(CUES))
        x[CUES.index("label")] = (1.0 if y else -1.0) * 0.5 + rng.normal(0, 0.3)
        x[CUES.index("flap")] = (-1.0 if y else 1.0) * 1.0 + rng.normal(0, 0.3)  # misleading
        rows.append({"x": x.tolist(), "label": y})
    m = fit(rows)
    assert m["beta"]["flap"] < 0.5 < m["beta"]["label"], m["beta"]
    before, after = score(rows, None), score(rows, m)
    assert after["auc"] > before["auc"]


def test_fit_starts_from_hand_set_model_when_uninformative():
    """With no signal, the fit should stay near the hand-set weights (beta = 1)."""
    rng = np.random.default_rng(1)
    rows = [{"x": rng.normal(0, 1, len(CUES)).tolist(), "label": int(rng.random() < 0.5)} for _ in range(200)]
    m = fit(rows, prior_strength=500.0)   # a strong prior is needed when there is little signal
    assert all(abs(b - 1.0) < 0.15 for b in m["beta"].values()), m["beta"]
