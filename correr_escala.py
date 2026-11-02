# Estudio de ESCALA (preregistrado en COORD.md, 2026-11-01 19:26)
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import itertools
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

NIVELES_M = {33: (9, 16, 32), 69: (12, 34, 68)}
SEMILLA_SORTEO = 20261102
SEMILLAS = tuple(range(5001, 5031))
BRAZOS = ("A", "B", "D")
PRESUPUESTO = 20000
PERFILES_CSV = "datos/perfiles_Arequipa.csv"
OBJETIVOS = "van_perdidas"
SEMILLA_LIMITES, N_MUESTRA_LIMITES = 777, 20000
REF_HV = 1.1
UMBRAL_NIVEL_ALTO = 1e-3          # HV_ref - HV < umbral
N_BOOT = 10000
N_PROCESOS = os.cpu_count() or 1

V2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2))
RES, FIG = V2 / "resultados", V2 / "figuras"

from fvopt.instancia import crear_instancia, leer_sitios, con_cambios, CALIBRADA   # noqa: E402
from fvopt.evaluador import Evaluador                                              # noqa: E402
from algoritmos.nsga2 import ejecutar, ParamNSGA                                   # noqa: E402
from algoritmos.espacial import OperadorEspacial                                   # noqa: E402
from algoritmos.metricas import no_dominados, normalizar, hv2d, igd_plus          # noqa: E402
from algoritmos.correr_estudio import (param_espacial, a12, holm, e_frac, curva_en_rejilla,  # noqa: E402
                                       NSGA, COLORES, NOMBRES)

_INST = {}


def conjuntos(n):
    """Barras candidatas anidadas por nivel de m (1-based)."""
    s, _ = leer_sitios(n)
    orig = sorted(int(b) for b in s.Barra)
    resto = sorted(set(range(2, n + 1)) - set(orig))
    extra = list(np.random.default_rng(SEMILLA_SORTEO).permutation(resto))
    out = {}
    for m in NIVELES_M[n]:
        out[m] = sorted(orig + [int(b) for b in extra[:m - len(orig)]])
    return out


def instancia(n, m):
    if (n, m) in _INST:
        return _INST[(n, m)]
    base = crear_instancia(n, perfiles_csv=V2 / PERFILES_CSV)
    s, tope = leer_sitios(n)
    orig = {int(b): i for i, b in enumerate(s.Barra)}
    esc = CALIBRADA[n]["escala_tope"]
    t_med = float(np.median(tope))
    cf_med = float(np.median(s.Costo_conexion_fijo_USD))
    cv_med = float(np.median(s.Costo_conexion_variable_USD_kWp))
    barras = np.array(conjuntos(n)[m], int)
    cap, cf, cv = [], [], []
    for b in barras:
        if b in orig:
            i = orig[b]
            cap.append(base.cap_kwp[list(base.barras).index(b)])
            cf.append(float(s.Costo_conexion_fijo_USD.iloc[i]))
            cv.append(float(s.Costo_conexion_variable_USD_kWp.iloc[i]))
        else:
            cap.append(float(np.round(t_med * esc, 0)))
            cf.append(cf_med)
            cv.append(cv_med)
    cap = np.array(cap)
    k = base.k_max
    inst = con_cambios(base, barras=barras, cap_kwp=cap, costo_fijo=np.array(cf), costo_var=np.array(cv),
                       factor_sitio=np.ones(len(barras)), p_total_max=float(np.sort(cap)[-k:].sum()))
    _INST[(n, m)] = inst
    return inst


def leer_json(nombre):
    return json.loads((RES / nombre).read_text())


def escribir_json(nombre, obj):
    (RES / nombre).write_text(json.dumps(obj, indent=1, default=float))


def una_corrida(t):
    n, m, brazo, semilla, esp, lim = t
    inst = instancia(n, m)
    ev = Evaluador(inst, objetivos=OBJETIVOS)
    op = None if brazo == "A" else OperadorEspacial(inst, param_espacial(brazo, esp))
    r = ejecutar(ev, semilla, ParamNSGA(presupuesto=PRESUPUESTO, **NSGA), np.array(lim["ideal"]),
                 np.array(lim["nadir"]), op, REF_HV)
    r.update(red=n, m=m, brazo=brazo, semilla=semilla)
    return r


def correr(tareas, desc):
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(N_PROCESOS) as pool:
        for i, r in enumerate(pool.map(una_corrida, tareas, chunksize=1)):
            out.append(r)
            if (i + 1) % max(1, len(tareas) // 20) == 0:
                print(f"  {desc}: {i + 1}/{len(tareas)}  {time.time() - t0:.0f} s", flush=True)
    print(f"{desc}: {len(tareas)} corridas en {time.time() - t0:.0f} s (pared)", flush=True)
    return out, time.time() - t0


def fase_sitios():
    filas = []
    for n in NIVELES_M:
        base = crear_instancia(n, perfiles_csv=V2 / PERFILES_CSV)
        for m in NIVELES_M[n]:
            inst = instancia(n, m)
            if m == NIVELES_M[n][0]:
                assert np.array_equal(inst.barras, base.barras) and np.allclose(inst.cap_kwp, base.cap_kwp)
                assert np.isclose(inst.p_total_max, base.p_total_max)
            ev = Evaluador(inst, objetivos=OBJETIVOS)
            r = ev.evaluar_lote(ev.aleatorio(np.random.default_rng(20261101), 2000))
            filas.append(dict(red=n, m=m, barras=" ".join(map(str, inst.barras)),
                              cap_kwp=" ".join(f"{c:.0f}" for c in inst.cap_kwp),
                              frac_inviable=float(np.mean([x.cv > 0 for x in r])),
                              barras_sin_carga=int((inst.red.p_kw[inst.barras - 1] == 0).sum())))
            print(filas[-1])
    pd.DataFrame(filas).to_csv(RES / "escala_sitios.csv", index=False)


def fase_limites():
    lim = {}
    for n in NIVELES_M:
        for m in NIVELES_M[n]:
            ev = Evaluador(instancia(n, m), objetivos=OBJETIVOS)
            X = ev.aleatorio(np.random.default_rng(SEMILLA_LIMITES), N_MUESTRA_LIMITES)
            X[0] = 0
            res = ev.evaluar_lote(X)
            F = np.array([r.f for r in res])
            ok = np.array([r.cv <= 0 for r in res])
            nd = F[ok][no_dominados(F[ok])]
            lim[f"{n}_{m}"] = dict(ideal=F[ok].min(0).tolist(), nadir=nd.max(0).tolist(), n_factibles=int(ok.sum()))
            print(n, m, lim[f"{n}_{m}"])
    escribir_json("escala_limites.json", lim)


def fase_tiempo():
    lim = leer_json("escala_limites.json")
    esp = leer_json("alg_parametros.json")["espacial"]
    tareas = [(n, NIVELES_M[n][-1], b, 4999, esp, lim[f"{n}_{NIVELES_M[n][-1]}"]) for n in NIVELES_M for b in BRAZOS]
    out, seg = correr(tareas, "tiempo")
    cpu = {f"{r['red']}_{r['m']}_{r['brazo']}": r["cpu_s"] for r in out}
    por_red = {n: np.mean([r["cpu_s"] for r in out if r["red"] == n]) for n in NIVELES_M}
    total = sum(por_red[n] * len(SEMILLAS) * len(BRAZOS) * len(NIVELES_M[n]) for n in NIVELES_M) / N_PROCESOS
    info = dict(cpu_s=cpu, estimacion_total_h=total / 3600, n_procesos=N_PROCESOS,
                nota="semilla 4999 (solo cronometraje, no entra al análisis); estimación con el m máximo (cota superior)")
    escribir_json("escala_tiempo.json", info)
    print(info)


def fase_prueba():
    lim = leer_json("escala_limites.json")
    esp = leer_json("alg_parametros.json")["espacial"]
    # primero IEEE69 (más lento) para equilibrar la cola
    tareas = [(n, m, b, s, esp, lim[f"{n}_{m}"]) for n in (69, 33) for m in NIVELES_M[n] for s in SEMILLAS
              for b in BRAZOS]
    out, seg = correr(tareas, "escala")
    for r in out:   # reducir tamaño: curva y archivo bastan
        r.pop("memoria", None)
    with open(RES / "escala_corridas.pkl", "wb") as fh:
        pickle.dump(dict(corridas=out, segundos_pared=seg, n_procesos=N_PROCESOS), fh)


def fase_analisis():
    from scipy.stats import wilcoxon, spearmanr, binomtest
    lim = leer_json("escala_limites.json")
    d = pickle.load(open(RES / "escala_corridas.pkl", "rb"))
    corr = d["corridas"]
    refs, filas = {}, []
    for n in NIVELES_M:
        for m in NIVELES_M[n]:
            L = lim[f"{n}_{m}"]
            ideal, nadir = np.array(L["ideal"]), np.array(L["nadir"])
            sub = [r for r in corr if r["red"] == n and r["m"] == m]
            F = np.vstack([r["F_archivo"] for r in sub])
            if n == 33 and m == 9 and (RES / "alg_frente_enum_IEEE33.csv").exists():
                F = np.vstack([F, pd.read_csv(RES / "alg_frente_enum_IEEE33.csv").to_numpy()])
            R = F[no_dominados(F)]
            Rz = normalizar(R, ideal, nadir)
            refs[(n, m)] = dict(R=R, hv=hv2d(Rz, REF_HV), n=len(R))
            for r in sub:
                Z = normalizar(r["F_archivo"], ideal, nadir)
                filas.append(dict(red=n, m=m, brazo=r["brazo"], semilla=r["semilla"], hv=r["hv_final"],
                                  igd_plus=igd_plus(Z, Rz), e95_ganancia=e_frac(r["curva"], 0.95, base=0),
                                  e99_ganancia=e_frac(r["curva"], 0.99, base=0),
                                  nivel_alto=bool(refs[(n, m)]["hv"] - r["hv_final"] < UMBRAL_NIVEL_ALTO),
                                  cpu_s=r["cpu_s"], n_evaluaciones=r["n_evaluaciones"], cortado=r["cortado"],
                                  n_traslados=r.get("n_traslados", 0)))
    df = pd.DataFrame(filas)
    assert (df.n_evaluaciones == PRESUPUESTO).all()
    df.to_csv(RES / "escala_corridas.csv", index=False)
    escribir_json("escala_referencias.json", {f"{n}_{m}": dict(hv_ref=v["hv"], n_puntos=v["n"])
                                              for (n, m), v in refs.items()})
    rng = np.random.default_rng(0)
    sentido = {"hv": 1, "igd_plus": -1, "e95_ganancia": -1, "e99_ganancia": -1}
    est, tend = [], []
    for n in NIVELES_M:
        for met, sg in sentido.items():
            bloque = []
            for m in NIVELES_M[n]:
                P = df[(df.red == n) & (df.m == m)].pivot(index="semilla", columns="brazo", values=met)
                for o in ("A", "B"):
                    dif = (P["D"] - P[o]).to_numpy()
                    p = 1.0 if np.allclose(dif, 0) else float(wilcoxon(P["D"], P[o]).pvalue)
                    bs = np.median(rng.choice(dif, (N_BOOT, len(dif))), 1)
                    na = df[(df.red == n) & (df.m == m)].pivot(index="semilla", columns="brazo", values="nivel_alto")
                    bloque.append(dict(red=n, m=m, metrica=met, comparacion=f"D vs {o}", mediana_D=P["D"].median(),
                                       mediana_otro=P[o].median(), mediana_dif=float(np.median(dif)),
                                       ic95_inf=float(np.quantile(bs, .025)), ic95_sup=float(np.quantile(bs, .975)),
                                       p_wilcoxon=p, A12_D_mejor=a12(sg * P["D"], sg * P[o]),
                                       nivel_alto_D=int(na["D"].sum()), nivel_alto_otro=int(na[o].sum())))
            for b, ph in zip(bloque, holm([b["p_wilcoxon"] for b in bloque])):
                b["p_holm"] = ph
            est += bloque
        for o in ("A", "B"):
            for met in ("hv", "igd_plus"):
                sub = df[df.red == n]
                P = sub.pivot_table(index=["m", "semilla"], columns="brazo", values=met).reset_index()
                rho, p = spearmanr(P["m"], P["D"] - P[o])
                tend.append(dict(red=n, metrica=met, comparacion=f"D-{o}", spearman_rho=rho, p=p, n_pares=len(P)))
        # McNemar exacto del nivel alto, D vs A, por m (descriptivo)
        for m in NIVELES_M[n]:
            na = df[(df.red == n) & (df.m == m)].pivot(index="semilla", columns="brazo", values="nivel_alto")
            b_, c_ = int((na.D & ~na.A).sum()), int((~na.D & na.A).sum())
            tend.append(dict(red=n, metrica=f"nivel_alto m={m}", comparacion="D vs A (McNemar exacto)",
                             spearman_rho=np.nan, p=binomtest(b_, b_ + c_).pvalue if b_ + c_ else 1.0,
                             n_pares=b_ + c_))
    est, tend = pd.DataFrame(est), pd.DataFrame(tend)
    est.to_csv(RES / "escala_estadistica.csv", index=False)
    tend.to_csv(RES / "escala_tendencia.csv", index=False)
    df.groupby(["red", "m", "brazo"])[["hv", "igd_plus", "e95_ganancia", "e99_ganancia", "nivel_alto", "cpu_s"]]\
      .median(numeric_only=True).to_csv(RES / "escala_resumen.csv")
    pd.set_option("display.width", 250)
    print(est[est.metrica == "hv"][["red", "m", "comparacion", "mediana_D", "mediana_otro", "mediana_dif", "ic95_inf",
                                     "ic95_sup", "p_holm", "A12_D_mejor", "nivel_alto_D", "nivel_alto_otro"]].to_string())
    print(est[est.metrica != "hv"][["red", "m", "metrica", "comparacion", "mediana_dif", "p_holm", "A12_D_mejor"]].to_string())
    print(tend.to_string())
    print(df.groupby(["red", "m", "brazo"])[["hv", "nivel_alto", "cpu_s"]].agg(["median", "mean"]))
    figuras(df, est, corr, refs)


def figuras(df, est, corr, refs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    # 1) diferencia pareada vs m con IC95
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, n in zip(ax, NIVELES_M):
        for o, dx, col in (("A", -0.15, COLORES["A"]), ("B", 0.15, COLORES["B"])):
            e = est[(est.red == n) & (est.metrica == "hv") & (est.comparacion == f"D vs {o}")]
            x = np.arange(len(e)) + dx
            a.errorbar(x, e.mediana_dif * 1e4, yerr=[(e.mediana_dif - e.ic95_inf) * 1e4, (e.ic95_sup - e.mediana_dif) * 1e4],
                       fmt="o-", color=col, capsize=4, label=f"D − {o}")
            for xi, (_, r) in zip(x, e.iterrows()):
                a.annotate(f"p$_H$={r.p_holm:.2f}", (xi, r.ic95_sup * 1e4), xytext=(0, 4), textcoords="offset points",
                           ha="center", fontsize=6.5, color=col)
        a.axhline(0, color="0.3", lw=0.8)
        a.set_xticks(range(len(NIVELES_M[n])), [f"m={m}" for m in NIVELES_M[n]])
        a.set_ylabel("Mediana de la diferencia pareada en HV (×10$^{-4}$)")
        a.set_title(f"IEEE{n}: ventaja de D en HV vs sitios candidatos (IC95 bootstrap)", fontsize=9)
        a.legend(fontsize=8)
        a.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(FIG / "escala_hv_vs_m.png", dpi=300)
    plt.close(fig)
    # 2) frentes por m: referencia + corrida de HV mediano por brazo
    for n in NIVELES_M:
        fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
        for a, m in zip(ax, NIVELES_M[n]):
            R = refs[(n, m)]["R"]
            o = np.argsort(R[:, 0])
            a.plot(R[o, 0] * 1e-6, R[o, 1] * 1e-3, color="0.6", lw=3, label="Referencia (unión)", zorder=1)
            for b in BRAZOS:
                sub = df[(df.red == n) & (df.m == m) & (df.brazo == b)]
                s_med = sub.iloc[np.argsort(sub.hv.to_numpy())[len(sub) // 2]].semilla
                r = next(r for r in corr if (r["red"], r["m"], r["brazo"], r["semilla"]) == (n, m, b, s_med))
                a.scatter(r["F_archivo"][:, 0] * 1e-6, r["F_archivo"][:, 1] * 1e-3, s=3, color=COLORES[b], zorder=2,
                          label=NOMBRES[b])
            a.set_title(f"IEEE{n}, m={m}: corrida de HV mediano", fontsize=9)
            a.set_xlabel("−VAN [MUSD]")
            a.set_ylabel("Pérdidas [MWh/año]")
            a.legend(fontsize=7, markerscale=3)
        fig.tight_layout()
        fig.savefig(FIG / f"escala_frentes_IEEE{n}.png", dpi=300)
        plt.close(fig)
    # 3) mapa de barras en el frente para m máximo
    fig, ax = plt.subplots(2, 1, figsize=(11, 10))
    for a, n in zip(ax, NIVELES_M):
        m = NIVELES_M[n][-1]
        inst = instancia(n, m)
        red = inst.red
        hijos = [[] for _ in range(red.n)]
        for k in range(1, red.n):
            hijos[red.padre[k]].append(k)
        pos, cont = {}, [0]
        sys.setrecursionlimit(10000)

        def ubicar(k, prof):
            if not hijos[k]:
                pos[k] = (prof, cont[0])
                cont[0] += 1
                return pos[k][1]
            ys = [ubicar(h, prof + 1) for h in hijos[k]]
            pos[k] = (prof, float(np.mean(ys)))
            return pos[k][1]
        ubicar(0, 0)
        idx = {int(b): i for i, b in enumerate(inst.barras)}
        frec = {b: np.zeros(inst.m) for b in ("A", "D")}
        for b in frec:
            tot = 0
            for r in corr:
                if r["red"] == n and r["m"] == m and r["brazo"] == b:
                    for fen in r["fen_archivo"]:
                        for bb, _ in fen:
                            frec[b][idx[bb]] += 1.0 / max(1, len(fen))
                        tot += 1
            frec[b] /= max(tot, 1)
        for k in range(1, red.n):
            p0, p1 = pos[red.padre[k]], pos[k]
            a.plot([p0[0], p0[0], p1[0]], [p0[1], p1[1], p1[1]], color="0.75", lw=1, zorder=1)
        c = np.array([pos[b - 1] for b in inst.barras])
        sc = a.scatter(c[:, 0], c[:, 1], s=25 + 1500 * frec["D"], c=frec["D"], cmap="viridis", vmin=0,
                       vmax=max(frec["D"].max(), 1e-9), edgecolor="k", lw=0.5, zorder=3)
        orden = [i for i in np.argsort(-frec["D"])[:6] if frec["D"][i] >= 0.005]
        for j, i in enumerate(orden):              # solo las 6 barras más frecuentes, para legibilidad
            off = np.sqrt(25 + 1500 * frec["D"][i]) / 2 + 2
            arriba = j % 2 == 0
            a.annotate(f"{inst.barras[i]}", c[i], xytext=(0, off if arriba else -off), textcoords="offset points",
                       ha="center", va="bottom" if arriba else "top", fontsize=7, fontweight="bold", zorder=4)
        tabla = "barra: D | A\n" + "\n".join(f"{inst.barras[i]}: {frec['D'][i]:.2f} | {frec['A'][i]:.2f}"
                                               for i in orden)
        a.text(0.99, 0.97, tabla, transform=a.transAxes, ha="right", va="top", fontsize=7, family="monospace",
               bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        a.annotate("cabecera", pos[0], xytext=(-5, -12), textcoords="offset points", fontsize=7)
        fig.colorbar(sc, ax=a, label="fracción de masa en el frente (D)")
        a.set_title(f"IEEE{n}, m={m} (todas las barras candidatas): presencia en los frentes finales de D "
                    f"(30 semillas; se rotulan las 6 más frecuentes)", fontsize=9)
        a.set_xlabel("profundidad en el árbol (n.º de ramas desde la cabecera)")
        a.set_yticks([])
        a.grid(False)
        a.margins(x=0.06, y=0.14)
    fig.tight_layout()
    fig.savefig(FIG / "escala_mapa_barras.png", dpi=300)
    plt.close(fig)


FASES = {"sitios": fase_sitios, "limites": fase_limites, "tiempo": fase_tiempo, "prueba": fase_prueba,
         "analisis": fase_analisis}

if __name__ == "__main__":
    for f in sys.argv[1:] or list(FASES):
        print(f"=== {f} ===", flush=True)
        t = time.time()
        FASES[f]()
        print(f"=== {f}: {time.time() - t:.0f} s ===", flush=True)
