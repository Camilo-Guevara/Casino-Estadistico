"""Pruebas de las tasas esperadas de rareza y diagnóstico de muestras."""
import numpy as np
import pytest

from engine import Model
from diagnostics import expected_rare_rate, experiment_diagnostics


@pytest.mark.parametrize("confidence", [.90, .95, .99])
def test_normal_rare_rate_matches_alpha(confidence):
    assert expected_rare_rate(Model("Normal"), confidence) == pytest.approx(1 - confidence)


@pytest.mark.parametrize("model", [
    Model(n=8, p=.2),
    Model(n=100, p=.5),
    Model("Poisson", lam=.3),
    Model("Poisson", lam=100),
])
def test_discrete_rare_rate_conservative(model):
    for confidence in [.90, .95, .99]:
        expected = expected_rare_rate(model, confidence)
        assert 0 <= expected <= 1 - confidence + 1e-10


@pytest.mark.parametrize("p", [0, 1])
def test_deterministic_binomial_has_no_rare_events_or_mean_z(p):
    m = Model(n=20, p=p)
    a = m.draw(np.random.default_rng(123), 100)
    r = experiment_diagnostics(m, a)
    assert expected_rare_rate(m) == 0
    assert r["rare_count"] == 0
    assert r["mean_z"] is None
    assert r["mean_standard_error"] == 0


def test_diagnostics_matches_manually_computed_mean():
    m = Model("Normal", mu=5, sigma=2)
    a = np.array([4.0, 5.0, 6.0, 7.0])
    r = experiment_diagnostics(m, a)
    assert r["size"] == 4
    assert r["observed_mean"] == pytest.approx(5.5)
    assert r["theoretical_mean"] == pytest.approx(5)
    assert r["mean_standard_error"] == pytest.approx(1)
    assert r["mean_z"] == pytest.approx(.5)
    assert r["expected_rare_count"] == pytest.approx(.2)
    assert r["rare_count"] == 0


def test_empty_and_invalid_input():
    m = Model()
    assert experiment_diagnostics(m, []) is None
    with pytest.raises(ValueError):
        experiment_diagnostics(m, [[1, 2]])
    with pytest.raises(ValueError):
        experiment_diagnostics(m, [float("nan")])
    with pytest.raises(ValueError):
        expected_rare_rate(m, 1)
