# Estudio completo
from __future__ import annotations

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import itertools
import json
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

REDES = (33, 69)
OBJETIVOS = "van_perdidas"
PERFILES_CSV = "datos/perfiles_Arequipa.csv"
CAMBIOS_INSTANCIA = {}                # p.ej. {"export_max_kw": 500}
CAMBIOS_ECON = {}
ETIQUETA = ""                         # sufijo de archivos para escenarios de sensibilidad
BRAZOS = ("A", "B", "C", "D")
SEMILLAS_CALIBRACION = (9001, 9002, 9003, 9004, 9005)
SEMILLAS_PRUEBA = tuple(range(1, 31))
SEMILLA_LIMITES = 777                 # muestra para ideal/nadir a priori
N_MUESTRA_LIMITES = 20000
REF_HV = 1.1                          # punto de referencia del HV en el espacio normalizado
NSGA = dict(pob=100, pc=0.9, eta_c=15.0, eta_m=20.0, pm=None)     # valores estándar de Deb (no calibrados)
PRESUPUESTO_PILOTO = 30000
PRESUPUESTO_MIN, PRESUPUESTO_MAX = 5000, 20000
TOL_CONVERGENCIA = 0.9997
ESPACIAL_DEFECTO = dict(p_traslado=0.3, eps=0.2, w_prox=0.4, w_sitio=0.2, w_mem=0.4, lam_mem=0.2,
                        ventana=10, delta_hv=1e-4, k_refuerzo=5)
REJILLA = dict(p_traslado=(0.05, 0.15, 0.4),
               pesos=((0.6, 0.2, 0.2), (0.4, 0.2, 0.4), (0.2, 0.2, 0.6)),     # (w_prox, w_sitio, w_mem)
               eps=(0.1, 0.3))
N_BOOT = 10000
N_PROCESOS = os.cpu_count() or 1

V2 = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(V2))
RES = V2 / "resultados"
FIG = V2 / "figuras"
PRE = f"alg{ETIQUETA}"

from fvopt.instancia import crear_instancia            # noqa: E402
from fvopt.evaluador import Evaluador                  # noqa: E402
from fvopt.economia import ParamEconomicos             # noqa: E402
from algoritmos.nsga2 import ejecutar, ParamNSGA       # noqa: E402
from algoritmos.espacial import OperadorEspacial, ParamEspacial   # noqa: E402
from algoritmos.metricas import no_dominados, normalizar, hv2d, igd_plus   # noqa: E402

_INST = {}


def instancia(n):
    if n not in _INST:
        econ = replace(ParamEconomicos(), **CAMBIOS_ECON) if CAMBIOS_ECON else None
        _INST[n] = crear_instancia(n, perfiles_csv=(V2 / PERFILES_CSV) if PERFILES_CSV else None, econ=econ,
                                   **CAMBIOS_INSTANCIA)
    return _INST[n]


def param_espacial(brazo, esp: dict):
    """Parámetros del operador para cada brazo (B y C = D con un componente apagado)."""
    e = dict(esp)
    e.setdefault("p_traslado_est", min(0.9, 2 * e["p_traslado"]))
    e.setdefault("eps_est", min(1.0, 2 * e["eps"]))
    if brazo == "B":
        e["destino"], e["usar_memoria"] = "uniforme", False
    elif brazo == "C":
        e["destino"], e["usar_memoria"] = "hibrido", False
    else:
        e["destino"], e["usar_memoria"] = "hibrido", True
    return ParamEspacial(**e)


def una_corrida(tarea):
    n, brazo, semilla, presupuesto, esp, lim = tarea
    inst = instancia(n)
    ev = Evaluador(inst, objetivos=OBJETIVOS)
    op = None if brazo == "A" else OperadorEspacial(inst, param_espacial(brazo, esp))
    r = ejecutar(ev, semilla, ParamNSGA(presupuesto=presupuesto, **NSGA), np.array(lim["ideal"]),
                 np.array(lim["nadir"]), op, REF_HV)
    r.update(red=n, brazo=brazo, semilla=semilla, presupuesto=presupuesto, esp=esp)
    return r


def correr(tareas, desc):
    t0 = time.time()
    out = []
    with ProcessPoolExecutor(N_PROCESOS) as pool:
        for i, r in enumerate(pool.map(una_corrida, tareas, chunksize=1)):
            out.append(r)
            if (i + 1) % max(1, len(tareas) // 10) == 0:
                print(f"  {desc}: {i + 1}/{len(tareas)}  {time.time() - t0:.0f} s", flush=True)
    print(f"{desc}: {len(tareas)} corridas en {time.time() - t0:.0f} s (pared)", flush=True)
    return out, time.time() - t0


def leer_json(nombre):
    return json.loads((RES / nombre).read_text())


def escribir_json(nombre, obj):
    (RES / nombre).write_text(json.dumps(obj, indent=1, default=float))


def fase_limites():
    """Ideal/nadir a priori"""
    lim = {}
    for n in REDES:
        ev = Evaluador(instancia(n), objetivos=OBJETIVOS)
        X = ev.aleatorio(np.random.default_rng(SEMILLA_LIMITES), N_MUESTRA_LIMITES)
        X[0] = 0
        res = ev.evaluar_lote(X)
        F = np.array([r.f for r in res])
        ok = np.array([r.cv <= 0 for r in res])
        Ff = F[ok]
        nd = Ff[no_dominados(Ff)]
        lim[str(n)] = dict(ideal=Ff.min(0).tolist(), nadir=nd.max(0).tolist(), n_muestra=N_MUESTRA_LIMITES,
                           n_factibles=int(ok.sum()), semilla=SEMILLA_LIMITES, objetivos=OBJETIVOS)
        print(n, lim[str(n)])
    escribir_json(f"{PRE}_limites.json", lim)


def curva_en_rejilla(curva, rejilla):
    e, h = curva[:, 0], curva[:, 1]
    idx = np.searchsorted(e, rejilla, side="right") - 1
    return np.where(idx >= 0, h[np.clip(idx, 0, None)], np.nan)


def fase_piloto():
    lim = leer_json(f"{PRE}_limites.json")
    tareas = [(n, b, s, PRESUPUESTO_PILOTO, ESPACIAL_DEFECTO, lim[str(n)])
              for n in REDES for b in ("A", "D") for s in SEMILLAS_CALIBRACION]
    out, seg = correr(tareas, "piloto")
    rej = np.arange(1000, PRESUPUESTO_PILOTO + 1, 1000)
    filas, elegido = [], {}
    for n in REDES:
        for b in ("A", "D"):
            H = np.array([curva_en_rejilla(r["curva"], rej) for r in out if r["red"] == n and r["brazo"] == b])
            med = np.median(H, 0)
            for e, v in zip(rej, med):
                filas.append(dict(red=n, brazo=b, evaluaciones=e, hv_mediano=v, frac_final=v / med[-1]))
            elegido[f"{n}{b}"] = int(rej[np.argmax(med >= TOL_CONVERGENCIA * med[-1])])
    pd.DataFrame(filas).to_csv(RES / f"{PRE}_piloto_curva.csv", index=False)
    pres = int(np.clip(max(elegido.values()), PRESUPUESTO_MIN, PRESUPUESTO_MAX))
    info = dict(presupuesto=pres, evaluaciones_99_9=elegido, tolerancia=TOL_CONVERGENCIA,
                presupuesto_piloto=PRESUPUESTO_PILOTO, segundos_pared=seg,
                cpu_medio_s={f"{n}": float(np.mean([r["cpu_s"] for r in out if r["red"] == n])) for n in REDES})
    escribir_json(f"{PRE}_presupuesto.json", info)
    print(info)


def fase_calibracion():
    lim = leer_json(f"{PRE}_limites.json")
    pres = leer_json(f"{PRE}_presupuesto.json")["presupuesto"]
    configs = []
    for p, w, e in itertools.product(REJILLA["p_traslado"], REJILLA["pesos"], REJILLA["eps"]):
        c = dict(ESPACIAL_DEFECTO)
        c.update(p_traslado=p, eps=e, w_prox=w[0], w_sitio=w[1], w_mem=w[2])
        configs.append(c)
    tareas = [(n, "D", s, pres, c, lim[str(n)]) for c in configs for n in REDES for s in SEMILLAS_CALIBRACION]
    out, seg = correr(tareas, "calibración")
    df = pd.DataFrame([dict(config=configs.index(r["esp"]), red=r["red"], semilla=r["semilla"],
                            hv=r["hv_final"], auc=float(np.mean(r["curva"][:, 1])), **r["esp"]) for r in out])
    # criterio: rango medio del HV final dentro de cada bloque (red, semilla); desempate: AUC media
    df["rango"] = df.groupby(["red", "semilla"]).hv.rank(ascending=False)
    res = df.groupby("config").agg(rango_medio=("rango", "mean"), hv_medio=("hv", "mean"),
                                   auc_media=("auc", "mean")).reset_index()
    res = res.sort_values(["rango_medio", "auc_media"], ascending=[True, False])
    df.to_csv(RES / f"{PRE}_calibracion_corridas.csv", index=False)
    res.merge(pd.DataFrame(configs).reset_index().rename(columns={"index": "config"})).to_csv(
        RES / f"{PRE}_calibracion_resumen.csv", index=False)
    mejor = configs[int(res.iloc[0].config)]
    escribir_json(f"{PRE}_parametros.json", dict(espacial=mejor, criterio="rango medio de HV final por (red, semilla); "
                  "desempate AUC", semillas=SEMILLAS_CALIBRACION, presupuesto=pres, segundos_pared=seg,
                  nsga=NSGA))
    print("mejor:", mejor)


def fase_prueba():
    lim = leer_json(f"{PRE}_limites.json")
    pres = leer_json(f"{PRE}_presupuesto.json")["presupuesto"]
    esp = leer_json(f"{PRE}_parametros.json")["espacial"]
    tareas = [(n, b, s, pres, esp, lim[str(n)]) for n in REDES for s in SEMILLAS_PRUEBA for b in BRAZOS]
    out, seg = correr(tareas, "prueba")
    with open(RES / f"{PRE}_corridas_prueba.pkl", "wb") as fh:
        pickle.dump(dict(corridas=out, segundos_pared=seg, n_procesos=N_PROCESOS), fh)


def fase_referencia():
    """IEEE33"""
    from fvopt.frente_exacto import enumerar, _objetivos
    kw = dict(CAMBIOS_INSTANCIA)
    if PERFILES_CSV:
        kw["perfiles_csv"] = V2 / PERFILES_CSV
    if CAMBIOS_ECON:
        kw["econ"] = replace(ParamEconomicos(), **CAMBIOS_ECON)
    inst, ev, sol, info = enumerar(33, objetivos=OBJETIVOS, **kw)
    F = _objetivos(sol, OBJETIVOS)
    nd = no_dominados(F)
    pd.DataFrame(F[nd], columns=["f1", "f2"]).to_csv(RES / f"{PRE}_frente_enum_IEEE33.csv", index=False)
    info.update(n_puntos=len(nd), objetivos=OBJETIVOS, perfiles=str(PERFILES_CSV))
    escribir_json(f"{PRE}_frente_enum_IEEE33_info.json", info)
    print(info)


def frente_referencia(n, corridas, lim):
    F = np.vstack([r["F_archivo"] for r in corridas if r["red"] == n])
    arch = RES / f"{PRE}_frente_enum_IEEE33.csv"
    if n == 33 and arch.exists():
        # la enumeración usa rejilla de 25 kWp: los algoritmos (resolución 1 kWp) la dominan
        E = pd.read_csv(arch).to_numpy()
        U = np.vstack([E, F])
        nd = no_dominados(U)
        frac = float((nd < len(E)).mean())
        return U[nd], f"enumeración ∪ 120 corridas; {frac:.1%} de los puntos vienen de la enumeración"
    return F[no_dominados(F)], "unión no dominada de las 120 corridas de prueba"


def e_frac(curva, frac, base=None):
    """Evaluaciones para alcanzar frac*HV_final (base=None) o frac de la ganancia HV_final-HV_0."""
    e, h = curva[:, 0], curva[:, 1]
    meta = frac * h[-1] if base is None else h[0] + frac * (h[-1] - h[0])
    return float(e[np.argmax(h >= meta - 1e-15)])


def a12(x, y):
    """Vargha-Delaney A12 = P(X > Y) + 0,5 P(X = Y)."""
    x, y = np.asarray(x)[:, None], np.asarray(y)[None, :]
    return float((x > y).mean() + 0.5 * (x == y).mean())


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    acum = 0.0
    for k, i in enumerate(o):
        acum = max(acum, (len(p) - k) * p[i])
        adj[i] = min(1.0, acum)
    return adj


def fase_analisis():
    from scipy.stats import wilcoxon
    lim = leer_json(f"{PRE}_limites.json")
    pres = leer_json(f"{PRE}_presupuesto.json")["presupuesto"]
    d = pickle.load(open(RES / f"{PRE}_corridas_prueba.pkl", "rb"))
    corr = d["corridas"]
    refs, filas = {}, []
    for n in REDES:
        L = lim[str(n)]
        ideal, nadir = np.array(L["ideal"]), np.array(L["nadir"])
        R, desc = frente_referencia(n, corr, L)
        Rz = normalizar(R, ideal, nadir)
        refs[n] = dict(R=R, desc=desc, hv=hv2d(Rz, REF_HV), n=len(R))
        for r in corr:
            if r["red"] != n:
                continue
            Z = normalizar(r["F_archivo"], ideal, nadir)
            filas.append(dict(red=n, brazo=r["brazo"], semilla=r["semilla"], hv=r["hv_final"],
                              igd_plus=igd_plus(Z, Rz), n_archivo=len(Z), e95=e_frac(r["curva"], 0.95),
                              e95_ganancia=e_frac(r["curva"], 0.95, base=0),
                              e99_ganancia=e_frac(r["curva"], 0.99, base=0),
                              cpu_s=r["cpu_s"], generaciones=r["generaciones"], n_llamadas=r["n_llamadas"],
                              n_evaluaciones=r["n_evaluaciones"], cortado=r["cortado"],
                              n_traslados=r.get("n_traslados", 0), n_activaciones=r.get("n_activaciones", 0),
                              gens_estancado=r.get("gens_estancado", 0)))
    df = pd.DataFrame(filas)
    assert (df.n_evaluaciones == pres).all(), "presupuesto desigual"
    df.to_csv(RES / f"{PRE}_prueba_corridas.csv", index=False)
    escribir_json(f"{PRE}_referencias.json", {str(n): dict(desc=v["desc"], hv_ref=v["hv"], n_puntos=v["n"])
                                              for n, v in refs.items()})
    q = lambda s: f"{s.median():.5g} [{s.quantile(.25):.5g}; {s.quantile(.75):.5g}]"
    met = ["hv", "igd_plus", "e95", "e95_ganancia", "e99_ganancia", "cpu_s", "n_traslados", "n_activaciones"]
    desc = df.groupby(["red", "brazo"])[met].agg(["median", lambda s: s.quantile(.25), lambda s: s.quantile(.75)])
    desc.columns = [f"{m}_{k}" for m, k in itertools.product(met, ["mediana", "q1", "q3"])]
    desc.reset_index().to_csv(RES / f"{PRE}_prueba_resumen.csv", index=False)
    rng = np.random.default_rng(0)
    est = []
    sentido = {"hv": 1, "igd_plus": -1, "e95": -1, "e95_ganancia": -1, "e99_ganancia": -1}   # +1: mayor es mejor
    for n in REDES:
        for m, sg in sentido.items():
            P = df[df.red == n].pivot(index="semilla", columns="brazo", values=m)
            bloque = []
            for o in ("A", "B", "C"):
                dif = (P["D"] - P[o]).to_numpy()
                if np.allclose(dif, 0):
                    p = 1.0
                else:
                    p = float(wilcoxon(P["D"], P[o], zero_method="wilcox", alternative="two-sided").pvalue)
                bs = np.median(rng.choice(dif, (N_BOOT, len(dif)), replace=True), 1)
                bloque.append(dict(red=n, metrica=m, comparacion=f"D vs {o}", n=len(dif),
                                   mediana_D=P["D"].median(), mediana_otro=P[o].median(),
                                   mediana_dif=float(np.median(dif)), ic95_inf=float(np.quantile(bs, .025)),
                                   ic95_sup=float(np.quantile(bs, .975)), p_wilcoxon=p,
                                   A12_D_mejor=a12(sg * P["D"], sg * P[o]),
                                   victorias_D=int((sg * dif > 0).sum()), empates=int((dif == 0).sum())))
            for b, ph in zip(bloque, holm([b["p_wilcoxon"] for b in bloque])):
                b["p_holm"] = ph
            est += bloque
    est = pd.DataFrame(est)
    est.to_csv(RES / f"{PRE}_prueba_estadistica.csv", index=False)
    pd.set_option("display.width", 250)
    print(est[["red", "metrica", "comparacion", "mediana_D", "mediana_otro", "mediana_dif", "ic95_inf",
               "ic95_sup", "p_holm", "A12_D_mejor", "victorias_D"]].to_string())
    print(df.groupby(["red", "brazo"])[["hv", "igd_plus", "e95", "e95_ganancia", "cpu_s", "n_traslados",
                                         "n_activaciones"]].median())
    print({n: (v["desc"], v["hv"], v["n"]) for n, v in refs.items()})
    figuras(df, corr, refs, lim, pres)


COLORES = {"A": "#4C72B0", "B": "#DD8452", "C": "#55A868", "D": "#C44E52"}
NOMBRES = {"A": "A: convencional", "B": "B: +traslado uniforme", "C": "C: +prox.+sitio",
           "D": "D: híbrido completo"}


def figuras(df, corr, refs, lim, pres):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    FIG.mkdir(exist_ok=True)
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3})
    lab2 = {"van_perdidas": ("Pérdidas [MWh/año]", 1e-3), "van_co2": ("−CO$_2$ evitado [t/año]", 1),
            "van_inversion": ("Inversión [MUSD]", 1e-6)}[OBJETIVOS]

    # 1) frentes: corrida de HV mediano por brazo + referencia
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, n in zip(ax, REDES):
        R = refs[n]["R"]
        o = np.argsort(R[:, 0])
        a.plot(R[o, 0] * 1e-6, R[o, 1] * lab2[1], color="0.6", lw=3, label="Referencia", zorder=1)
        for b in BRAZOS:
            sub = df[(df.red == n) & (df.brazo == b)]
            s_med = sub.iloc[np.argsort(sub.hv.to_numpy())[len(sub) // 2]].semilla
            r = next(r for r in corr if r["red"] == n and r["brazo"] == b and r["semilla"] == s_med)
            F = r["F_archivo"]
            a.scatter(F[:, 0] * 1e-6, F[:, 1] * lab2[1], s=3, color=COLORES[b], label=NOMBRES[b], zorder=2)
        a.set_title(f"IEEE{n}: corrida de HV mediano por brazo ({pres} eval.)", fontsize=9)
        a.set_xlabel("−VAN [MUSD]")
        a.set_ylabel(lab2[0])
        a.legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    fig.savefig(FIG / f"{PRE}_frentes.png", dpi=300)
    plt.close(fig)

    # 2) curvas HV (mediana + IQR) y déficit respecto de la referencia (log)
    rej = np.arange(100, pres + 1, 100)
    fig, ax = plt.subplots(2, 2, figsize=(10, 6.5), sharex="col")
    for j, n in enumerate(REDES):
        for b in BRAZOS:
            H = np.array([curva_en_rejilla(r["curva"], rej) for r in corr if r["red"] == n and r["brazo"] == b])
            q1, med, q3 = np.nanquantile(H, [.25, .5, .75], axis=0)
            ax[0, j].plot(rej, med, color=COLORES[b], label=NOMBRES[b], lw=1.2)
            ax[0, j].fill_between(rej, q1, q3, color=COLORES[b], alpha=0.2, lw=0)
            dq = refs[n]["hv"] - np.array([q3, med, q1])
            ax[1, j].plot(rej, np.maximum(dq[1], 1e-6), color=COLORES[b], lw=1.2)
            ax[1, j].fill_between(rej, np.maximum(dq[0], 1e-6), np.maximum(dq[2], 1e-6), color=COLORES[b],
                                  alpha=0.2, lw=0)
        ax[0, j].axhline(refs[n]["hv"], color="0.4", ls="--", lw=0.8, label="HV de la referencia")
        ax[0, j].set_title(f"IEEE{n}")
        ax[0, j].set_ylabel("HV normalizado")
        hv0 = np.nanmedian([r["curva"][0, 1] for r in corr if r["red"] == n])
        ax[0, j].set_ylim(hv0 - 0.005, refs[n]["hv"] + 0.005)
        ax[1, j].set_yscale("log")
        ax[1, j].set_ylabel("HV$_{ref}$ − HV (log)")
        ax[1, j].set_xlabel("Evaluaciones reales (n_evaluaciones)")
        ax[0, j].legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(FIG / f"{PRE}_curvas_hv.png", dpi=300)
    plt.close(fig)

    # 3) boxplots HV e IGD+
    fig, ax = plt.subplots(2, 2, figsize=(9, 6))
    for j, n in enumerate(REDES):
        for i, (m, t) in enumerate([("hv", "HV normalizado (mayor es mejor)"), ("igd_plus", "IGD+ (menor es mejor)")]):
            datos = [df[(df.red == n) & (df.brazo == b)][m].to_numpy() for b in BRAZOS]
            bp = ax[i, j].boxplot(datos, patch_artist=True, widths=0.6)
            for p, b in zip(bp["boxes"], BRAZOS):
                p.set_facecolor(COLORES[b])
                p.set_alpha(0.6)
            for k, v in enumerate(datos):
                ax[i, j].scatter(np.full(len(v), k + 1) + np.random.default_rng(k).uniform(-.15, .15, len(v)), v,
                                 s=5, color="k", zorder=3)
            ax[i, j].set_xticks(range(1, 5), list(BRAZOS))
            ax[i, j].set_title(f"IEEE{n}: {t}", fontsize=9)
            ax[i, j].ticklabel_format(axis="y", style="sci", scilimits=(-3, 3))
    fig.tight_layout()
    fig.savefig(FIG / f"{PRE}_boxplots.png", dpi=300)
    plt.close(fig)

    # 4) frecuencia de barras en el frente sobre el árbol (unión de archivos de D, cada solución masa 1)
    fig, ax = plt.subplots(2, 1, figsize=(11, 10))
    for a, n in zip(ax, REDES):
        inst = instancia(n)
        red = inst.red
        hijos = [[] for _ in range(red.n)]
        for k in range(1, red.n):
            hijos[red.padre[k]].append(k)
        pos, cont = {}, [0]
        def ubicar(k, prof):
            if not hijos[k]:
                pos[k] = (prof, cont[0]); cont[0] += 1
                return pos[k][1]
            ys = [ubicar(h, prof + 1) for h in hijos[k]]
            pos[k] = (prof, float(np.mean(ys)))
            return pos[k][1]
        sys.setrecursionlimit(10000)
        ubicar(0, 0)
        frec = {"A": np.zeros(inst.m), "D": np.zeros(inst.m)}
        idx = {int(b): i for i, b in enumerate(inst.barras)}
        for b in ("A", "D"):
            tot = 0
            for r in corr:
                if r["red"] == n and r["brazo"] == b:
                    for fen in r["fen_archivo"]:
                        for bb, _ in fen:
                            frec[b][idx[bb]] += 1.0 / max(1, len(fen))
                        tot += 1
            frec[b] /= max(tot, 1)
        for k in range(1, red.n):
            p0, p1 = pos[red.padre[k]], pos[k]
            a.plot([p0[0], p0[0], p1[0]], [p0[1], p1[1], p1[1]], color="0.75", lw=1, zorder=1)
        xy = np.array([pos[k] for k in range(red.n)])
        a.scatter(xy[:, 0], xy[:, 1], s=8, color="0.5", zorder=2)
        c = np.array([pos[b - 1] for b in inst.barras])
        sc = a.scatter(c[:, 0], c[:, 1], s=60 + 1500 * frec["D"], c=frec["D"], cmap="viridis", vmin=0,
                       vmax=max(frec["D"].max(), 1e-9), edgecolor="k", zorder=3)
        for k, ((x, y), b, fa, fd) in enumerate(zip(c, inst.barras, frec["A"], frec["D"])):
            off = np.sqrt(60 + 1500 * fd) / 2 + 3            # radio del marcador en puntos + holgura
            arriba = (k % 2 == 0) != (int(b) == 68 and n == 69)     # 68 arriba para no tapar la barra 12
            a.annotate(f"{b}\nD {fd:.2f} | A {fa:.2f}", (x, y), xytext=(0, off if arriba else -off),
                       textcoords="offset points", ha="center", va="bottom" if arriba else "top", fontsize=6.5,
                       bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=4)
        a.annotate("cabecera", xy[0], xytext=(-5, -12), textcoords="offset points", fontsize=7)
        fig.colorbar(sc, ax=a, label="fracción de masa en el frente (D)")
        a.set_title(f"IEEE{n}: presencia de barras candidatas en los frentes finales (30 semillas; masa 1 por solución)",
                    fontsize=9)
        a.margins(x=0.06, y=0.14)
        a.set_xlabel("profundidad en el árbol (n.º de ramas desde la cabecera)")
        a.set_yticks([])
        a.grid(False)
    fig.tight_layout()
    fig.savefig(FIG / f"{PRE}_mapa_barras.png", dpi=300)
    plt.close(fig)


FASES = {"limites": fase_limites, "referencia": fase_referencia, "piloto": fase_piloto, "calibracion": fase_calibracion,
         "prueba": fase_prueba, "analisis": fase_analisis}

if __name__ == "__main__":
    fases = sys.argv[1:] or list(FASES)
    for f in fases:
        print(f"=== {f} ===", flush=True)
        t = time.time()
        FASES[f]()
        print(f"=== {f}: {time.time() - t:.0f} s ===", flush=True)
