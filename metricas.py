# Métricas biobjetivo (minimización)
from __future__ import annotations

import numpy as np


def no_dominados(F: np.ndarray) -> np.ndarray:
    """Índices del conjunto no dominado (2 objetivos, minimización), sin duplicados."""
    F = np.asarray(F, float)
    if len(F) == 0:
        return np.zeros(0, int)
    orden = np.lexsort((F[:, 1], F[:, 0]))       # por f1, luego f2
    sel, mejor_f2 = [], np.inf
    for i in orden:
        if F[i, 1] < mejor_f2 - 1e-12:
            sel.append(i)
            mejor_f2 = F[i, 1]
    return np.array(sel, int)


def normalizar(F, ideal, nadir):
    return (np.asarray(F, float) - ideal) / (np.asarray(nadir) - np.asarray(ideal))


def hv2d(Z: np.ndarray, ref: float = 1.1) -> float:
    """Hipervolumen 2D de puntos normalizados respecto de (ref, ref), dividido por ref^2."""
    Z = np.asarray(Z, float)
    if len(Z) == 0:
        return 0.0
    Z = Z[(Z[:, 0] < ref) & (Z[:, 1] < ref)]
    if len(Z) == 0:
        return 0.0
    Z = Z[no_dominados(Z)]
    Z = Z[np.argsort(Z[:, 0])]
    f1 = np.append(Z[1:, 0], ref)
    return float(((f1 - Z[:, 0]) * (ref - Z[:, 1])).sum() / ref ** 2)


def igd_plus(Z: np.ndarray, R: np.ndarray) -> float:
    """IGD+ (Ishibuchi et al."""
    Z, R = np.asarray(Z, float), np.asarray(R, float)
    if len(Z) == 0:
        return np.inf
    d = np.maximum(Z[None, :, :] - R[:, None, :], 0.0)        # (|R|, |Z|, 2)
    return float(np.sqrt((d ** 2).sum(2)).min(1).mean())
