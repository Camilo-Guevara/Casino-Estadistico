"""Motor estadístico puro: sin UI, red ni estado global."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy import stats

@dataclass(frozen=True)
class Model:
    kind: str = 'Binomial'
    n: int = 100
    p: float = .3
    lam: float = 4.
    mu: float = 0.
    sigma: float = 1.

    def __post_init__(self):
        if self.kind not in ('Binomial', 'Poisson', 'Normal'):
            raise ValueError('Modelo desconocido')
        if not isinstance(self.n, int) or not 1 <= self.n <= 500:
            raise ValueError('n debe ser entero entre 1 y 500')
        if not np.isfinite(self.p) or not 0 <= self.p <= 1:
            raise ValueError('p debe estar entre 0 y 1')
        if not np.isfinite(self.lam) or not .1 <= self.lam <= 100:
            raise ValueError('lambda fuera de rango')
        if not np.isfinite(self.mu) or not np.isfinite(self.sigma) or self.sigma <= 0:
            raise ValueError('Normal requiere media finita y sigma positiva')

    @property
    def distribution(self):
        if self.kind == 'Binomial': return stats.binom(self.n, self.p)
        if self.kind == 'Poisson': return stats.poisson(self.lam)
        return stats.norm(self.mu, self.sigma)

    @property
    def moments(self):
        d = self.distribution
        return float(d.mean()), float(d.var())

    def draw(self, rng, size):
        if not isinstance(size, int) or not 1 <= size <= 1000:
            raise ValueError('El lote debe contener de 1 a 1000 resultados')
        if self.kind == 'Binomial': return rng.binomial(self.n, self.p, size)
        if self.kind == 'Poisson': return rng.poisson(self.lam, size)
        return rng.normal(self.mu, self.sigma, size)

    def tail_probability(self, x):
        """Colas bilaterales iguales: min(1,2*min(P[X<=x],P[X>=x])).
        Discreta conservadora, no la prueba por orden de masa de binomtest.
        """
        x = np.asarray(x)
        d = self.distribution
        upper = d.sf(x) if self.kind == 'Normal' else d.sf(x - 1)
        return np.minimum(1., 2 * np.minimum(d.cdf(x), upper))

    def zscore(self, x):
        mean, var = self.moments
        return (np.asarray(x) - mean) / np.sqrt(var) if var > 0 else np.full(np.asarray(x).shape, np.nan)

    def grid(self, observed=()):
        d = self.distribution
        lo, hi = d.ppf(.0001), d.ppf(.9999)
        if len(observed): lo, hi = min(lo, min(observed)), max(hi, max(observed))
        if self.kind == 'Binomial':
            lo, hi = max(0, lo), min(self.n, hi)
        if self.kind == 'Normal':
            x = np.linspace(lo, hi, 400)
            return x, d.pdf(x)
        x = np.arange(int(lo), int(hi) + 1)
        return x, d.pmf(x)


def mean_inference(model, values, confidence=.95):
    """IC de media y contraste bilateral bajo el modelo iid especificado."""
    if not 0 < confidence < 1: raise ValueError('Confianza inválida')
    a = np.asarray(values)
    if not len(a): return None
    m, v = model.moments
    size = len(a)
    total = float(a.sum())
    alpha = 1 - confidence
    if model.kind == 'Binomial':
        trials = size * model.n
        test = stats.binomtest(int(total), trials, model.p)
        ci = test.proportion_ci(confidence_level=confidence, method='exact')
        lower, upper = model.n * ci.low, model.n * ci.high
        pvalue, method = test.pvalue, 'Binomial exacta; IC Clopper–Pearson para p, escalado por n'
    elif model.kind == 'Poisson':
        lower = 0. if total == 0 else stats.chi2.ppf(alpha / 2, 2 * total) / (2 * size)
        upper = stats.chi2.ppf(1 - alpha / 2, 2 * (total + 1)) / (2 * size)
        d = stats.poisson(size * model.lam)
        pvalue = min(1., 2 * min(d.cdf(total), d.sf(total - 1)))
        method = 'Poisson: colas iguales; IC exacto de Garwood para lambda'
    else:
        se = model.sigma / np.sqrt(size)
        z = (a.mean() - model.mu) / se
        critical = stats.norm.ppf(1 - alpha / 2)
        lower, upper = a.mean() - critical * se, a.mean() + critical * se
        pvalue, method = 2 * stats.norm.sf(abs(z)), 'Prueba Z e IC de media con sigma conocida'
    return dict(lower=float(lower), upper=float(upper), pvalue=float(pvalue), method=method,
                null_mean=m, confidence=confidence, sample_size=size)


def export_frame(model, values):
    import pandas as pd
    a = np.asarray(values)
    f = pd.DataFrame({'tirada': np.arange(1, len(a)+1), 'resultado': a,
                      'z_score': model.zscore(a), 'probabilidad_colas': model.tail_probability(a),
                      'media_acumulada': np.cumsum(a)/np.arange(1,len(a)+1)})
    for k, v in asdict(model).items(): f['modelo_'+k] = v
    return f
