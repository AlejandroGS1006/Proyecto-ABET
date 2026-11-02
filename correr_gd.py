# Estudio GD con tope regulatorio 200 kW AC por planta (preregistrado en COORD.md, 2026-11-0
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REDES = {33: 32, 69: 68}                 # red -> m (todas las barras no slack)
NIVELES_K = (3, 6, 10, 15)
TOPE_PLANTA_KWP = 200.0 * 1.2            # 200 kW AC x DC/AC 1,2 (placa del inversor)
SEMILLAS = tuple(range(6001, 6031))
SEMILLA_TIEMPO = 5999
BRAZOS = ("A", "B", "D")
PRESUPUESTO = 20000
OBJETIVOS = "van_perdidas"
SEMILLA_LIMITES, N_MUESTRA_LIMITES = 777, 20000
REF_HV = 1.1
N_BOOT = 10000
UMBRAL_NO_APLANA = 1e-3                  # ganancia de HV mediano en las últimas 5000 eval.
N_PROCESOS = 2

V2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2))
RES, FIG = V2 / "resultados", V2 / "figuras"

from fvopt.instancia import con_cambios                                            # noqa: E402
from fvopt.evaluador import Evaluador                                              # noqa: E402
from algoritmos.nsga2 import ejecutar, ParamNSGA                                   # noqa: E402
from algoritmos.espacial import OperadorEspacial                                   # noqa: E402
from algoritmos.metricas import no_dominados, normalizar, hv2d, igd_plus          # noqa: E402
from algoritmos.correr_estudio import (param_espacial, a12, holm, e_frac, curva_en_rejilla,  # noqa: E402
                                       NSGA, COLORES, NOMBRES)
from algoritmos import correr_escala as ESC                                        # noqa: E402

_INST = {}


def instancia(n, K):
    if (n, K) not in _INST:
        base = ESC.instancia(n, REDES[n])
        cap = np.minimum(base.cap_kwp, TOPE_PLANTA_KWP)
        _INST[(n, K)] = con_cambios(base, cap_kwp=cap, k_max=K, p_total_max=float(np.sort(cap)[-K:].sum()))
    return _INST[(n, K)]


def leer_json(nombre):
    return json.loads((RES / nombre).read_text())


def escribir_json(nombre, obj):
    (RES / nombre).write_text(json.dumps(obj, indent=1, default=float))


def una_corrida(t):
    n, K, brazo, semilla, esp, lim = t
    inst = instancia(n, K)
    ev = Evaluador(inst, objetivos=OBJETIVOS)
    op = None if brazo == "A" else OperadorEspacial(inst, param_espacial(brazo, esp))
    r = ejecutar(ev, semilla, ParamNSGA(presupuesto=PRESUPUESTO, **NSGA), np.array(lim["ideal"]),
                 np.array(lim["nadir"]), op, REF_HV)
    r.pop("memoria", None)
    r.update(red=n, K=K, brazo=brazo, semilla=semilla)
    return r


def correr(tareas, desc):
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(N_PROCESOS) as pool:
        for i, r in enumerate(pool.map(una_corrida, tareas, chunksize=1)):
            out.append(r)
            if (i + 1) % max(1, len(tareas) // 24) == 0:
                print(f"  {desc}: {i + 1}/{len(tareas)}  {time.time() - t0:.0f} s", flush=True)
    print(f"{desc}: {len(tareas)} corridas en {time.time() - t0:.0f} s (pared)", flush=True)
    return out, time.time() - t0


def fase_sitios():
    filas = []
    for n in REDES:
        for K in NIVELES_K:
            inst = instancia(n, K)
            ev = Evaluador(inst, objetivos=OBJETIVOS)
            res = ev.evaluar_lote(ev.aleatorio(np.random.default_rng(20261101), 2000))
            inv = [r for r in res if r.cv > 0]
            act = {k: float(np.mean([r.detalle["cv"][k] > 0 for r in res])) for k in res[0].detalle["cv"]}
            filas.append(dict(red=n, K=K, m=inst.m, tope_kwp_min=inst.cap_kwp.min(), tope_kwp_max=inst.cap_kwp.max(),
                              p_total_max=inst.p_total_max, frac_inviable=len(inv) / len(res),
                              **{f"activa_{k}": v for k, v in act.items()},
                              p_media_kwp=float(np.mean([r.detalle["p_total_kwp"] for r in res]))))
            print(filas[-1])
    pd.DataFrame(filas).to_csv(RES / "gd_sitios.csv", index=False)


def fase_limites():
    lim = {}
    for n in REDES:
        for K in NIVELES_K:
            ev = Evaluador(instancia(n, K), objetivos=OBJETIVOS)
            X = ev.aleatorio(np.random.default_rng(SEMILLA_LIMITES), N_MUESTRA_LIMITES)
            X[0] = 0
            res = ev.evaluar_lote(X)
            F = np.array([r.f for r in res])
            ok = np.array([r.cv <= 0 for r in res])
            nd = F[ok][no_dominados(F[ok])]
            lim[f"{n}_{K}"] = dict(ideal=F[ok].min(0).tolist(), nadir=nd.max(0).tolist(), n_factibles=int(ok.sum()))
            print(n, K, lim[f"{n}_{K}"])
    escribir_json("gd_limites.json", lim)


def fase_tiempo():
    lim = leer_json("gd_limites.json")
    esp = leer_json("alg_parametros.json")["espacial"]
    tareas = [(n, 15, b, SEMILLA_TIEMPO, esp, lim[f"{n}_15"]) for n in REDES for b in BRAZOS]
    out, seg = correr(tareas, "tiempo")
    por_red = {n: float(np.mean([r["cpu_s"] for r in out if r["red"] == n])) for n in REDES}
    total = sum(por_red[n] * len(SEMILLAS) * len(BRAZOS) * len(NIVELES_K) for n in REDES) / N_PROCESOS
    info = dict(cpu_s={f"{r['red']}_{r['brazo']}": r["cpu_s"] for r in out}, pared_s=seg,
                estimacion_total_h=total / 3600, nota="K=15 (cota superior), semilla 5999 fuera del análisis")
    escribir_json("gd_tiempo.json", info)
    print(info)


def fase_prueba():
    lim = leer_json("gd_limites.json")
    esp = leer_json("alg_parametros.json")["espacial"]
    tareas = [(n, K, b, s, esp, lim[f"{n}_{K}"]) for n in (69, 33) for K in NIVELES_K[::-1] for s in SEMILLAS
              for b in BRAZOS]
    out, seg = correr(tareas, "gd")
    with open(RES / "gd_corridas.pkl", "wb") as fh:
        pickle.dump(dict(corridas=out, segundos_pared=seg, n_procesos=N_PROCESOS), fh)


def fase_analisis():
    from scipy.stats import wilcoxon, spearmanr
    lim = leer_json("gd_limites.json")
    corr = pickle.load(open(RES / "gd_corridas.pkl", "rb"))["corridas"]
    refs, filas, extremos = {}, [], []
    for n in REDES:
        for K in NIVELES_K:
            L = lim[f"{n}_{K}"]
            ideal, nadir = np.array(L["ideal"]), np.array(L["nadir"])
            sub = [r for r in corr if r["red"] == n and r["K"] == K]
            F = np.vstack([r["F_archivo"] for r in sub])
            R = F[no_dominados(F)]
            Rz = normalizar(R, ideal, nadir)
            refs[(n, K)] = dict(R=R, hv=hv2d(Rz, REF_HV), n=len(R))
            ev = Evaluador(instancia(n, K), objetivos=OBJETIVOS)
            for r in sub:
                Z = normalizar(r["F_archivo"], ideal, nadir)
                ptot = np.array([sum(p for _, p in f) for f in r["fen_archivo"]])
                npl = np.array([len(f) for f in r["fen_archivo"]])
                c = r["curva"]
                hv15 = curva_en_rejilla(c, np.array([PRESUPUESTO - 5000]))[0]
                filas.append(dict(red=n, K=K, brazo=r["brazo"], semilla=r["semilla"], hv=r["hv_final"],
                                  igd_plus=igd_plus(Z, Rz), e95_ganancia=e_frac(c, 0.95, base=0),
                                  e99_ganancia=e_frac(c, 0.99, base=0), ganancia_ult5000=r["hv_final"] - hv15,
                                  p_max_kwp=float(ptot.max()), n_plantas_medio=float(npl.mean()),
                                  n_plantas_max=int(npl.max()), n_archivo=len(Z), cpu_s=r["cpu_s"],
                                  n_evaluaciones=r["n_evaluaciones"], cortado=r["cortado"]))
                # márgenes de la solución de máxima penetración
                fen = r["fen_archivo"][int(ptot.argmax())]
                det = ev.indicadores(dict(fen))["resultado"].detalle
                extremos.append(dict(red=n, K=K, brazo=r["brazo"], semilla=r["semilla"], p_kwp=det["p_total_kwp"],
                                     n_plantas=det["n_plantas"], vmax=det["vmax"], i_util_max=det["i_util_max"],
                                     export_max_kw=det["export_max_kw"], VAN_usd=det["VAN_usd"],
                                     perdidas_kwh=det["perdidas_kwh"]))
    df = pd.DataFrame(filas)
    assert (df.n_evaluaciones == PRESUPUESTO).all()
    df.to_csv(RES / "gd_corridas.csv", index=False)
    ext = pd.DataFrame(extremos)
    ext.to_csv(RES / "gd_extremos.csv", index=False)
    escribir_json("gd_referencias.json", {f"{n}_{K}": dict(hv_ref=v["hv"], n_puntos=v["n"]) for (n, K), v in refs.items()})
    rng = np.random.default_rng(0)
    sentido = {"hv": 1, "igd_plus": -1, "e95_ganancia": -1, "e99_ganancia": -1}
    est, tend = [], []
    for n in REDES:
        for met, sg in sentido.items():
            bloque = []
            for K in NIVELES_K:
                P = df[(df.red == n) & (df.K == K)].pivot(index="semilla", columns="brazo", values=met)
                for o in ("A", "B"):
                    dif = (P["D"] - P[o]).to_numpy()
                    p = 1.0 if np.allclose(dif, 0) else float(wilcoxon(P["D"], P[o]).pvalue)
                    bs = np.median(rng.choice(dif, (N_BOOT, len(dif))), 1)
                    bloque.append(dict(red=n, K=K, metrica=met, comparacion=f"D vs {o}", mediana_D=P["D"].median(),
                                       mediana_otro=P[o].median(), mediana_dif=float(np.median(dif)),
                                       ic95_inf=float(np.quantile(bs, .025)), ic95_sup=float(np.quantile(bs, .975)),
                                       p_wilcoxon=p, A12_D_mejor=a12(sg * P["D"], sg * P[o])))
            for b, ph in zip(bloque, holm([b["p_wilcoxon"] for b in bloque])):
                b["p_holm"] = ph
            est += bloque
        for o in ("A", "B"):
            for met in ("hv", "igd_plus"):
                P = df[df.red == n].pivot_table(index=["K", "semilla"], columns="brazo", values=met).reset_index()
                rho, p = spearmanr(P["K"], P["D"] - P[o])
                tend.append(dict(red=n, metrica=met, comparacion=f"D-{o}", spearman_rho=rho, p=p, n_pares=len(P)))
    est, tend = pd.DataFrame(est), pd.DataFrame(tend)
    est.to_csv(RES / "gd_estadistica.csv", index=False)
    tend.to_csv(RES / "gd_tendencia.csv", index=False)
    # H2
    h2 = {}
    for n in REDES:
        e = est[(est.red == n) & (est.metrica == "hv") & (est.comparacion == "D vs A") & (est.K == NIVELES_K[-1])].iloc[0]
        t = tend[(tend.red == n) & (tend.metrica == "hv") & (tend.comparacion == "D-A")].iloc[0]
        h2[str(n)] = dict(p_holm_Kmax=float(e.p_holm), mediana_dif_Kmax=float(e.mediana_dif), rho=float(t.spearman_rho),
                          p_rho=float(t.p), cumple=bool(e.p_holm < 0.05 and e.mediana_dif > 0 and t.spearman_rho > 0
                                                       and t.p < 0.05))
    # concentración del frente (masa en las 4 barras más frecuentes, por brazo y K)
    conc = []
    for n in REDES:
        for K in NIVELES_K:
            inst = instancia(n, K)
            idx = {int(b): i for i, b in enumerate(inst.barras)}
            for b in BRAZOS:
                fr = np.zeros(inst.m)
                tot = 0
                for r in corr:
                    if (r["red"], r["K"], r["brazo"]) == (n, K, b):
                        for fen in r["fen_archivo"]:
                            for bb, _ in fen:
                                fr[idx[bb]] += 1.0 / max(1, len(fen))
                            tot += 1
                fr /= max(tot, 1)
                top = np.argsort(-fr)[:4]
                conc.append(dict(red=n, K=K, brazo=b, masa_top4=float(fr[top].sum()),
                                 top4=" ".join(str(inst.barras[i]) for i in top),
                                 prof_media_ponderada=float(fr @ (inst.red.A[inst.barras - 1].sum(1)))))
    conc = pd.DataFrame(conc)
    conc.to_csv(RES / "gd_concentracion.csv", index=False)
    res = df.groupby(["red", "K", "brazo"])[["hv", "igd_plus", "e95_ganancia", "e99_ganancia", "ganancia_ult5000",
                                             "p_max_kwp", "n_plantas_medio", "n_plantas_max", "cpu_s"]].median()
    res.to_csv(RES / "gd_resumen.csv")
    escribir_json("gd_h2.json", h2)
    pd.set_option("display.width", 250)
    print(est[est.metrica == "hv"][["red", "K", "comparacion", "mediana_D", "mediana_otro", "mediana_dif", "ic95_inf",
                                     "ic95_sup", "p_holm", "A12_D_mejor"]].to_string())
    print(est[est.metrica != "hv"][["red", "K", "metrica", "comparacion", "mediana_dif", "p_holm", "A12_D_mejor"]].to_string())
    print(tend.to_string())
    print(res.to_string())
    print(ext.groupby(["red", "K"])[["p_kwp", "n_plantas", "vmax", "i_util_max", "export_max_kw"]].median())
    print(conc.to_string())
    print(h2)
    figuras(df, est, corr, refs)


def figuras(df, est, corr, refs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    # 1) diferencia pareada vs K
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for a, n in zip(ax, REDES):
        for o, dx in (("A", -0.15), ("B", 0.15)):
            e = est[(est.red == n) & (est.metrica == "hv") & (est.comparacion == f"D vs {o}")]
            x = np.arange(len(e)) + dx
            a.errorbar(x, e.mediana_dif * 1e4, yerr=[(e.mediana_dif - e.ic95_inf) * 1e4, (e.ic95_sup - e.mediana_dif) * 1e4],
                       fmt="o-", color=COLORES[o], capsize=4, label=f"D − {o}")
            for xi, (_, r) in zip(x, e.iterrows()):
                a.annotate(f"p$_H$={r.p_holm:.2f}", (xi, r.ic95_sup * 1e4), xytext=(0, 4), textcoords="offset points",
                           ha="center", fontsize=6.5, color=COLORES[o])
        a.axhline(0, color="0.3", lw=0.8)
        a.set_xticks(range(len(NIVELES_K)), [f"K={k}" for k in NIVELES_K])
        a.set_ylabel("Mediana de la diferencia pareada en HV (×10$^{-4}$)")
        a.set_title(f"IEEE{n} (m={REDES[n]}, tope 240 kWp/planta): ventaja de D vs K (IC95)", fontsize=9)
        a.legend(fontsize=8)
        a.margins(y=0.15)
    fig.tight_layout()
    fig.savefig(FIG / "gd_hv_vs_k.png", dpi=300)
    plt.close(fig)
    # 2) frentes por K
    for n in REDES:
        fig, ax = plt.subplots(1, 4, figsize=(16, 3.8))
        for a, K in zip(ax, NIVELES_K):
            R = refs[(n, K)]["R"]
            o = np.argsort(R[:, 0])
            a.plot(R[o, 0] * 1e-6, R[o, 1] * 1e-3, color="0.6", lw=3, label="Referencia (unión)", zorder=1)
            for b in BRAZOS:
                sub = df[(df.red == n) & (df.K == K) & (df.brazo == b)]
                s_med = sub.iloc[np.argsort(sub.hv.to_numpy())[len(sub) // 2]].semilla
                r = next(r for r in corr if (r["red"], r["K"], r["brazo"], r["semilla"]) == (n, K, b, s_med))
                a.scatter(r["F_archivo"][:, 0] * 1e-6, r["F_archivo"][:, 1] * 1e-3, s=3, color=COLORES[b], zorder=2,
                          label=NOMBRES[b])
            a.set_title(f"IEEE{n}, K={K}: corrida de HV mediano", fontsize=9)
            a.set_xlabel("−VAN [MUSD]")
            a.set_ylabel("Pérdidas [MWh/año]")
            a.legend(fontsize=7, markerscale=3)
        fig.tight_layout()
        fig.savefig(FIG / f"gd_frentes_IEEE{n}.png", dpi=300)
        plt.close(fig)
    # 3) mapa de barras, K máximo
    K = NIVELES_K[-1]
    fig, ax = plt.subplots(2, 1, figsize=(11, 10))
    for a, n in zip(ax, REDES):
        inst = instancia(n, K)
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
                if (r["red"], r["K"], r["brazo"]) == (n, K, b):
                    for fen in r["fen_archivo"]:
                        for bb, _ in fen:
                            frec[b][idx[bb]] += 1.0 / max(1, len(fen))
                        tot += 1
            frec[b] /= max(tot, 1)
        for k in range(1, red.n):
            p0, p1 = pos[red.padre[k]], pos[k]
            a.plot([p0[0], p0[0], p1[0]], [p0[1], p1[1], p1[1]], color="0.75", lw=1, zorder=1)
        c = np.array([pos[b - 1] for b in inst.barras])
        esc = 1500 / max(frec["D"].max(), 1e-9) * 0.3
        sc = a.scatter(c[:, 0], c[:, 1], s=25 + esc * frec["D"], c=frec["D"], cmap="viridis", vmin=0,
                       vmax=max(frec["D"].max(), 1e-9), edgecolor="k", lw=0.5, zorder=3)
        orden = [i for i in np.argsort(-frec["D"])[:8] if frec["D"][i] >= 0.005]
        for j, i in enumerate(orden):
            off = np.sqrt(25 + esc * frec["D"][i]) / 2 + 2
            arriba = j % 2 == 0
            a.annotate(f"{inst.barras[i]}", c[i], xytext=(0, off if arriba else -off), textcoords="offset points",
                       ha="center", va="bottom" if arriba else "top", fontsize=7, fontweight="bold", zorder=4)
        tabla = "barra: D | A\n" + "\n".join(f"{inst.barras[i]}: {frec['D'][i]:.2f} | {frec['A'][i]:.2f}" for i in orden)
        a.text(0.99, 0.97, tabla, transform=a.transAxes, ha="right", va="top", fontsize=7, family="monospace",
               bbox=dict(boxstyle="round", fc="white", ec="0.7"))
        a.annotate("cabecera", pos[0], xytext=(-5, -12), textcoords="offset points", fontsize=7)
        fig.colorbar(sc, ax=a, label="fracción de masa en el frente (D)")
        a.set_title(f"IEEE{n}, K={K}, tope 240 kWp/planta: presencia de barras en los frentes finales de D "
                    f"(30 semillas; se rotulan las 8 más frecuentes)", fontsize=9)
        a.set_xlabel("profundidad en el árbol (n.º de ramas desde la cabecera)")
        a.set_yticks([])
        a.grid(False)
        a.margins(x=0.06, y=0.14)
    fig.tight_layout()
    fig.savefig(FIG / "gd_mapa_barras.png", dpi=300)
    plt.close(fig)


FASES = {"sitios": fase_sitios, "limites": fase_limites, "tiempo": fase_tiempo, "prueba": fase_prueba,
         "analisis": fase_analisis}

if __name__ == "__main__":
    for f in sys.argv[1:] or list(FASES):
        print(f"=== {f} ===", flush=True)
        t = time.time()
        FASES[f]()
        print(f"=== {f}: {time.time() - t:.0f} s ===", flush=True)
