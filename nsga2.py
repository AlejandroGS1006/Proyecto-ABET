# NSGA-II (Deb et al.
from __future__ import annotations

from dataclasses import dataclass, field
from time import process_time

import numpy as np

from .metricas import no_dominados, normalizar, hv2d


@dataclass
class ParamNSGA:
    pob: int = 100
    pc: float = 0.9
    eta_c: float = 15.0
    eta_m: float = 20.0
    pm: float | None = None          # por defecto 1/D
    presupuesto: int = 10000
    max_llamadas_factor: int = 50


def rangos_restringidos(F, CV):
    """Rango con dominancia restringida de Deb"""
    N = len(F)
    rango = np.empty(N, int)
    fac = np.flatnonzero(CV <= 0)
    inf = np.flatnonzero(CV > 0)
    r = 0
    if len(fac):
        Ff = F[fac]
        le = (Ff[:, None, :] <= Ff[None, :, :]).all(2)
        lt = (Ff[:, None, :] < Ff[None, :, :]).any(2)
        dom = le & lt                                 # dom[i, j]: i domina a j
        n_dom = dom.sum(0)
        resto = np.ones(len(fac), bool)
        while resto.any():
            frente = resto & (n_dom == 0)
            rango[fac[frente]] = r
            resto &= ~frente
            n_dom = n_dom - dom[frente].sum(0)
            n_dom[~resto] = -1
            r += 1
    if len(inf):
        _, inv = np.unique(CV[inf], return_inverse=True)
        rango[inf] = r + inv
    return rango


def crowding(F, rango):
    N = len(F)
    d = np.zeros(N)
    for r in np.unique(rango):
        idx = np.flatnonzero(rango == r)
        if len(idx) <= 2:
            d[idx] = np.inf
            continue
        for k in range(F.shape[1]):
            o = idx[np.argsort(F[idx, k], kind="stable")]
            rngk = F[o[-1], k] - F[o[0], k]
            d[o[0]] = d[o[-1]] = np.inf
            if rngk > 0:
                d[o[1:-1]] += (F[o[2:], k] - F[o[:-2], k]) / rngk
    return d


def torneo(rango, crowd, rng, n):
    a, b = rng.integers(len(rango), size=(2, n))
    gana_a = (rango[a] < rango[b]) | ((rango[a] == rango[b]) & (crowd[a] > crowd[b]))
    empate = (rango[a] == rango[b]) & (crowd[a] == crowd[b])
    gana_a = np.where(empate, rng.random(n) < 0.5, gana_a)
    return np.where(gana_a, a, b)


def sbx(P1, P2, lo, hi, pc, eta, rng):
    C1, C2 = P1.copy(), P2.copy()
    N, D = P1.shape
    cruza = rng.random(N) < pc
    for i in np.flatnonzero(cruza):
        for j in range(D):
            if rng.random() > 0.5:
                continue
            y1, y2 = sorted((P1[i, j], P2[i, j]))
            if y2 - y1 < 1e-14:
                continue
            yl, yu = lo[j], hi[j]
            rnd = rng.random()
            def hijo(beta):
                alpha = 2.0 - beta ** -(eta + 1)
                if rnd <= 1.0 / alpha:
                    bq = (rnd * alpha) ** (1.0 / (eta + 1))
                else:
                    bq = (1.0 / (2.0 - rnd * alpha)) ** (1.0 / (eta + 1))
                return bq
            b1 = 1.0 + 2.0 * (y1 - yl) / (y2 - y1)
            c1 = 0.5 * (y1 + y2 - hijo(b1) * (y2 - y1))
            b2 = 1.0 + 2.0 * (yu - y2) / (y2 - y1)
            c2 = 0.5 * (y1 + y2 + hijo(b2) * (y2 - y1))
            c1, c2 = np.clip(c1, yl, yu), np.clip(c2, yl, yu)
            if rng.random() < 0.5:
                c1, c2 = c2, c1
            C1[i, j], C2[i, j] = c1, c2
    return C1, C2


def mutacion_polinomica(X, lo, hi, pm, eta, rng):
    X = X.copy()
    mask = rng.random(X.shape) < pm
    for i, j in zip(*np.nonzero(mask)):
        y, yl, yu = X[i, j], lo[j], hi[j]
        d1, d2 = (y - yl) / (yu - yl), (yu - y) / (yu - yl)
        r = rng.random()
        mp = 1.0 / (eta + 1)
        if r < 0.5:
            v = 2 * r + (1 - 2 * r) * (1 - d1) ** (eta + 1)
            dq = v ** mp - 1
        else:
            v = 2 * (1 - r) + 2 * (r - 0.5) * (1 - d2) ** (eta + 1)
            dq = 1 - v ** mp
        X[i, j] = np.clip(y + dq * (yu - yl), yl, yu)
    return X


class Archivo:
    """No dominados factibles de todo lo evaluado (sin tope de tamaño)."""
    def __init__(self):
        self.F = np.zeros((0, 2))
        self.fen = []

    def agregar(self, F, fens):
        if len(F) == 0:
            return
        Ft = np.vstack([self.F, F])
        ft = self.fen + list(fens)
        idx = no_dominados(Ft)
        self.F, self.fen = Ft[idx], [ft[i] for i in idx]


def poblacion_inicial(ev, n, semilla):
    """Población común a todos los brazos (depende solo de la semilla)"""
    rng = np.random.default_rng([int(semilla), 0])
    X = ev.aleatorio(rng, n)
    X[:, 0::2] += 0.5                                # centro de la celda entera
    X[0] = 0.0
    X[0, 0::2] = 0.5
    return X


def ejecutar(ev, semilla, par: ParamNSGA, ideal, nadir, operador=None, ref_hv=1.1):
    """Corre NSGA-II."""
    t0 = process_time()
    D = 2 * ev.K
    lo = np.zeros(D)
    hi = np.tile([ev.m + 1 - 1e-9, 1.0], ev.K)
    pm = par.pm or 1.0 / D
    rng = np.random.default_rng([int(semilla), 1])
    arch = Archivo()

    def evaluar(X):
        res = ev.evaluar_lote(X)
        F = np.array([r.f for r in res])
        CV = np.array([r.cv for r in res])
        fens = [r.fenotipo for r in res]
        ok = CV <= 0
        arch.agregar(F[ok], [f for f, o in zip(fens, ok) if o])
        return F, CV

    X = poblacion_inicial(ev, par.pob, semilla)
    F, CV = evaluar(X)
    curva = []

    def registrar():
        hv = hv2d(normalizar(arch.F, ideal, nadir), ref_hv) if len(arch.F) else 0.0
        curva.append((ev.n_evaluaciones, hv))
        return hv

    hv = registrar()
    if operador is not None:
        operador.actualizar(arch.fen, ev.inst.barras, hv)
    gen = 0
    cortado = False
    while ev.n_evaluaciones < par.presupuesto:
        if ev.n_llamadas > par.max_llamadas_factor * par.presupuesto:
            cortado = True
            break
        rango = rangos_restringidos(F, CV)
        cr = crowding(F, rango)
        n_par = (par.pob + 1) // 2
        i1 = torneo(rango, cr, rng, n_par)
        i2 = torneo(rango, cr, rng, n_par)
        C1, C2 = sbx(X[i1], X[i2], lo, hi, par.pc, par.eta_c, rng)
        H = np.vstack([C1, C2])[:par.pob]
        H = mutacion_polinomica(H, lo, hi, pm, par.eta_m, rng)
        if operador is not None:
            H = operador.aplicar(H, rng)
        # truncar para no exceder el presupuesto de evaluaciones reales
        restante = par.presupuesto - ev.n_evaluaciones
        nuevos, corte = set(), len(H)
        for i, h in enumerate(H):
            f = ev.decodificar(h)
            if f not in ev.cache and f not in nuevos:
                if len(nuevos) == restante:
                    corte = i
                    break
                nuevos.add(f)
        H = H[:corte]
        FH, CVH = evaluar(H)
        X, F, CV = np.vstack([X, H]), np.vstack([F, FH]), np.concatenate([CV, CVH])
        rango = rangos_restringidos(F, CV)
        cr = crowding(F, rango)
        orden = np.lexsort((-cr, rango))[:par.pob]
        X, F, CV = X[orden], F[orden], CV[orden]
        gen += 1
        hv = registrar()
        if operador is not None:
            operador.actualizar(arch.fen, ev.inst.barras, hv)
    out = dict(F_archivo=arch.F, fen_archivo=arch.fen, curva=np.array(curva), generaciones=gen,
               n_evaluaciones=ev.n_evaluaciones, n_llamadas=ev.n_llamadas, cortado=cortado,
               cpu_s=process_time() - t0, hv_final=hv)
    if operador is not None:
        out.update(n_traslados=operador.n_traslados, n_activaciones=operador.n_activaciones,
                   gens_estancado=operador.gens_estancado, memoria=operador.memoria.copy())
    return out
