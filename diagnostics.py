"""Diagnóstico pedagógico de calibración para los modelos del casino.

Distingue la frecuencia empírica de eventos raros de su frecuencia esperada
bajo la distribución elegida. No pretende diagnosticar sesgo con una única
muestra ni corregir la inspección repetida de p-valores.
"""
import numpy as np


def expected_rare_rate(model, confidence=.95):
    """P(colas bilaterales < alfa) bajo el modelo, no siempre igual a alfa."""
    if not 0 < confidence < 1:
        raise ValueError("La cobertura debe estar entre 0 y 1.")
    alpha = 1.0 - confidence
    if model.kind == "Normal":
        return float(alpha)
    if model.kind == "Binomial":
        support = np.arange(model.n + 1)
    else:
        # Poisson tiene soporte infinito: masa omitida <= aprox. 1e-12.
        upper = int(model.distribution.ppf(1 - 1e-12))
        support = np.arange(upper + 1)
    p = model.distribution.pmf(support)
    rare = model.tail_probability(support) < alpha
    return float(np.clip(p[rare].sum(), 0, 1))


def experiment_diagnostics(model, values, confidence=.95):
    """Devuelve indicadores observados vs. teóricos para una muestra fija.

    El estadístico de la media usa el error estándar teórico, no la desviación
    estándar de una sola tirada; no debe confundirse con el Z individual.
    """
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or (a.size and not np.isfinite(a).all()):
        raise ValueError("Se espera una muestra unidimensional y finita.")
    if not a.size:
        return None
    mean, variance = model.moments
    baseline = expected_rare_rate(model, confidence)
    alpha = 1 - confidence
    count = int(np.count_nonzero(model.tail_probability(a) < alpha))
    observed = count / a.size
    se = float(np.sqrt(variance / a.size))
    sample_mean = float(a.mean())
    return {
        "size": int(a.size),
        "theoretical_mean": float(mean),
        "observed_mean": sample_mean,
        "absolute_mean_error": abs(sample_mean - mean),
        "mean_standard_error": se,
        "mean_z": float((sample_mean - mean) / se) if se > 0 else None,
        "rare_count": count,
        "expected_rare_rate": baseline,
        "observed_rare_rate": float(observed),
        "expected_rare_count": float(a.size * baseline),
        "rare_rate_se": float(np.sqrt(baseline * (1 - baseline) / a.size)),
    }
