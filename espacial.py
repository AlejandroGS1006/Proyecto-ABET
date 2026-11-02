# Búsqueda espacial adaptativa híbrida
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def distancia_impedancia(inst) -> np.ndarray:
    """(m, m) distancia eléctrica entre sitios candidatos"""
    red = inst.red
    z = np.hypot(red.r_pu, red.x_pu)
    A = red.A[inst.barras - 1]                      # (m, n-1) ramas en el camino raíz -> sitio
    return np.abs(A[:, None, :] - A[None, :, :]) @ z


def matriz_proximidad(inst) -> np.ndarray:
    """Prox[o, d] = exp(-Z/mediana) con diagonal 0 (fila = origen)."""
    D = distancia_impedancia(inst)
    m = len(D)
    esc = np.median(D[np.triu_indices(m, 1)])
    W = np.exp(-D / esc)
    np.fill_diagonal(W, 0.0)
    return W


def puntaje_sitio(inst, pvout_rel=None, area_m2=None, exponente_area=1.0) -> np.ndarray:
    """Puntaje geográfico-solar por sitio (suma 1)."""
    pv = np.asarray(inst.factor_sitio if pvout_rel is None else pvout_rel, float)
    ar = np.asarray(inst.cap_kwp if area_m2 is None else area_m2, float)
    s = pv * (ar / ar.max()) ** exponente_area
    return s / s.sum()


@dataclass
class ParamEspacial:
    destino: str = "hibrido"         # 'uniforme' (brazo B) | 'hibrido' (C y D)
    usar_memoria: bool = True        # False en el brazo C
    p_traslado: float = 0.25         # probabilidad de traslado por hijo
    p_traslado_est: float = 0.6      # ... en modo estancado
    eps: float = 0.2                 # componente uniforme mínima
    eps_est: float = 0.5             # ... en modo estancado
    w_prox: float = 0.4
    w_sitio: float = 0.2
    w_mem: float = 0.4
    lam_mem: float = 0.2             # suavizado exponencial de la memoria
    ventana: int = 10                # generaciones de la ventana deslizante de HV
    delta_hv: float = 1e-4           # mejora mínima de HV normalizado en la ventana
    k_refuerzo: int = 5              # generaciones en modo reforzado antes de reiniciar


class OperadorEspacial:
    def __init__(self, inst, par: ParamEspacial, puntaje=None):
        self.par, self.m, self.K = par, inst.m, inst.k_max
        self.prox = matriz_proximidad(inst)
        self.puntaje = puntaje_sitio(inst) if puntaje is None else np.asarray(puntaje, float)
        self.memoria = np.full(self.m, 1.0 / self.m)
        self.hist_hv = []
        self.refuerzo_restante = 0
        self.n_traslados = 0
        self.n_activaciones = 0
        self.gens_estancado = 0

    @property
    def estancado(self):
        return self.refuerzo_restante > 0

    def actualizar(self, fenotipos_frente, barras, hv):
        """Llamar una vez por generación con el archivo no dominado factible y su HV."""
        p = self.par
        if p.usar_memoria and len(fenotipos_frente):
            idx = {int(b): i for i, b in enumerate(barras)}
            senal = np.zeros(self.m)
            n = 0
            for fen in fenotipos_frente:
                if len(fen) == 0:
                    continue
                for b, _ in fen:
                    senal[idx[b]] += 1.0 / len(fen)      # masa 1 por solución
                n += 1
            if n:
                self.memoria = (1 - p.lam_mem) * self.memoria + p.lam_mem * senal / n
        # detector de estancamiento con ventana deslizante y reinicio
        if self.refuerzo_restante > 0:
            self.refuerzo_restante -= 1
            self.gens_estancado += 1
            if self.refuerzo_restante == 0:
                self.hist_hv = []                        # reinicio: se exige una ventana nueva
        self.hist_hv.append(hv)
        if self.refuerzo_restante == 0 and len(self.hist_hv) > p.ventana:
            if self.hist_hv[-1] - self.hist_hv[-1 - p.ventana] < p.delta_hv:
                self.refuerzo_restante = p.k_refuerzo
                self.n_activaciones += 1

    def distribucion(self, origen: int, libres: np.ndarray) -> np.ndarray:
        """Probabilidades sobre `libres` (índices 0-based de sitio) desde `origen`."""
        p = self.par
        k = len(libres)
        u = np.full(k, 1.0 / k)
        if p.destino == "uniforme":
            return u
        def nrm(v):
            s = v.sum()
            return v / s if s > 0 else u
        wm = p.w_mem if p.usar_memoria else 0.0
        ws = np.array([p.w_prox, p.w_sitio, wm])
        ws = ws / ws.sum()
        guia = (ws[0] * nrm(self.prox[origen, libres]) + ws[1] * nrm(self.puntaje[libres])
                + ws[2] * nrm(self.memoria[libres]))
        eps = p.eps_est if self.estancado else p.eps
        return eps * u + (1 - eps) * guia

    def aplicar(self, X: np.ndarray, rng) -> np.ndarray:
        """Traslado sobre una matriz de hijos (N, 2K)."""
        p = self.par
        pt = p.p_traslado_est if self.estancado else p.p_traslado
        X = X.copy()
        for x in X:
            if rng.random() >= pt:
                continue
            sitios = np.clip(np.floor(x[0::2]), 0, self.m).astype(int)
            activos, vistos = [], set()
            for k, s in enumerate(sitios):
                if s > 0 and s not in vistos:
                    activos.append(k)
                    vistos.add(s)
            if not activos:
                continue
            k = activos[rng.integers(len(activos))]
            origen = sitios[k] - 1
            libres = np.array([j for j in range(self.m) if (j + 1) not in vistos])
            if len(libres) == 0:
                continue
            d = libres[rng.choice(len(libres), p=self.distribucion(origen, libres))]
            x[2 * k] = d + 1 + 0.5                       # centro de la celda del sitio d+1
            self.n_traslados += 1
        return X
