import numpy as np
import pytest

from drift_phish import DriftAwarePhishingDetector, FrameworkConfig
from drift_phish.drift import KSDriftMonitor
from drift_phish.eval import classification_report_dict
from drift_phish.features import (
    PAPER_FEATURE_NAMES,
    URL_FEATURE_NAMES,
    extract_matrix,
    extract_paper_features,
)
from drift_phish.learning import OnlineSGDClassifier
from drift_phish.selection import DynamicFeatureSelector


def make_data(n=3000, d=12, seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = (X[:, 0] + 0.8 * X[:, 1] - 0.6 * X[:, 2] > 0).astype(int)
    X = X.copy()
    X[:, :4] += shift
    return X, y


# --------------------------------------------------------------- features
def test_feature_names_unique_and_superset():
    assert len(set(PAPER_FEATURE_NAMES)) == len(PAPER_FEATURE_NAMES)
    assert set(URL_FEATURE_NAMES) <= set(PAPER_FEATURE_NAMES)
    assert len(PAPER_FEATURE_NAMES) >= 50


def test_paper_features_on_suspicious_url():
    f = extract_paper_features("http://secure-login.paypal.com.evil.tk/verify/account.php?redirect=http://x.com/a")
    assert f["scheme_http"] == 1.0
    assert f["has_login"] == 1.0 and f["has_verify"] == 1.0 and f["has_secure"] == 1.0
    assert f["suspicious_tld"] == 1.0
    assert f["brand_name_proximity"] >= 1.0
    assert f["domain_mismatch"] == 1.0
    assert len(f) == len(PAPER_FEATURE_NAMES)


def test_ip_and_empty_url():
    assert extract_paper_features("http://192.168.1.1/login")["is_ip"] == 1.0
    assert all(v == 0.0 for v in extract_paper_features("").values())
    X = extract_matrix(["https://example.com", "http://a.b/c"])
    assert X.shape == (2, len(PAPER_FEATURE_NAMES))


# -------------------------------------------------------------- KS monitor
def test_ks_default_delta_matches_paper():
    assert KSDriftMonitor(feature_names=["f"]).delta == 0.05


def test_ks_detects_shift_and_ignores_iid():
    rng = np.random.default_rng(0)
    mon = KSDriftMonitor(feature_names=["a", "b"], reference_size=1500)
    mon.set_reference(rng.normal(size=(1500, 2)))
    same = mon.test(rng.normal(size=(500, 2)))
    shifted = mon.test(rng.normal(loc=1.0, size=(500, 2)))
    assert not same.drifted.all()
    assert shifted.drifted.all()


# ---------------------------------------------------------------- selector
def test_selector_percentile_threshold_and_informative_features():
    X, y = make_data()
    names = [f"f{i}" for i in range(X.shape[1])]
    sel = DynamicFeatureSelector(feature_names=names).fit(X, y)
    assert sel.lasso_alpha == 1e-4 and sel.percentile == 75.0
    assert 3 <= sel.selected_.sum() <= 5  # top quartile of 12 features
    assert {"f0", "f1"} <= set(sel.selected_names)
    assert sel.transform(X).shape[1] == sel.selected_.sum()


def test_selector_update_reports_changes():
    X, y = make_data()
    names = [f"f{i}" for i in range(X.shape[1])]
    sel = DynamicFeatureSelector(feature_names=names).fit(X, y)
    rng = np.random.default_rng(1)
    X2 = rng.normal(size=(800, 12))
    y2 = (X2[:, 8] - X2[:, 9] > 0).astype(int)  # relevance moved to f8, f9
    info = sel.update(X2, y2, drifted=np.ones(12, bool))
    assert set(info) == {"obsolete", "removed", "added"}
    assert {"f8", "f9"} & set(sel.selected_names)


# ------------------------------------------------------------------ online
@pytest.mark.parametrize("loss", ["hinge", "log_loss", "perceptron"])
def test_online_sgd_variants_learn(loss):
    X, y = make_data()
    m = OnlineSGDClassifier(loss=loss).fit_offline(X[:2000], y[:2000])
    assert (m.predict(X[2000:]) == y[2000:]).mean() > 0.9
    m.partial_fit(X[2000:2500], y[2000:2500])


def test_mask_zeroes_dropped_features():
    X, y = make_data()
    m = OnlineSGDClassifier().fit_offline(X, y)
    mask = np.zeros(X.shape[1])
    mask[:3] = 1
    m.set_mask(mask)
    assert np.all(m.coef_[3:] == 0)
    s1 = m.decision_function(X[:5])
    X2 = X[:5].copy()
    X2[:, 5:] += 100
    assert np.allclose(s1, m.decision_function(X2))  # masked features have no effect


# --------------------------------------------------------------- framework
def test_metrics_positive_class_is_phishing():
    r = classification_report_dict(np.array([1, 1, 0, 0]), np.array([1, 0, 1, 0]))
    assert r["tp"] == 1 and r["fn"] == 1 and r["fp"] == 1 and r["tn"] == 1
    assert r["fpr"] == 0.5 and r["recall"] == 0.5


def test_table4_defaults():
    c = FrameworkConfig()
    assert (c.ks_delta, c.buffer_size, c.batch_size, c.eta0, c.lasso_alpha) == (0.05, 1500, 500, 0.01, 1e-4)
    assert c.loss == "hinge" and c.random_state == 42


def test_frozen_mode_never_changes_model():
    X, y = make_data(4000)
    names = [f"f{i}" for i in range(12)]
    det = DriftAwarePhishingDetector(names, FrameworkConfig(mode="frozen")).fit_offline(X[:2000], y[:2000])
    before = det.model.coef_.copy()
    det.process_stream(X[2000:], y[2000:])
    assert np.array_equal(before, det.model.coef_)


def test_adaptive_detects_drift_and_beats_frozen():
    names = [f"f{i}" for i in range(12)]
    Xtr, ytr = make_data(3000, seed=0)
    Xs, ys = make_data(4000, seed=1, shift=3.0)
    ys = (Xs[:, 4] - Xs[:, 5] > 0).astype(int)  # concept moves to unselected features
    results = {}
    for mode in ("frozen", "adaptive"):
        det = DriftAwarePhishingDetector(names, FrameworkConfig(mode=mode)).fit_offline(Xtr, ytr)
        out = det.process_stream(Xs, ys)
        results[mode] = (out["y_pred"] == out["y_true"]).mean()
        if mode == "adaptive":
            assert any(e["type"] == "drift" for e in det.events)
    assert results["adaptive"] > results["frozen"]


def test_periodic_batch_retrains():
    names = [f"f{i}" for i in range(12)]
    X, y = make_data(6000)
    det = DriftAwarePhishingDetector(
        names, FrameworkConfig(mode="periodic_batch", retrain_every=1000)
    ).fit_offline(X[:2000], y[:2000])
    det.process_stream(X[2000:], y[2000:])
    assert any(e["type"] == "batch_retrain" for e in det.events)
