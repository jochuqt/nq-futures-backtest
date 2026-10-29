"""Motor de las estrategias basadas en la literatura (propuestas con Claude) (estrategias\\Literatura\\*.toml).
Todas las horas son de Nueva York. In-sample = oct-2025 a mar-2026 (no lee meses posteriores).
Out-of-sample = abr-sep 2026: exige resultados\\Literatura\\criterios_oos.md escrito antes,
y cada archivo se corre UNA sola vez (queda en registro_oos.csv).

Uso:
    python -m scripts.backtest.backtest_literatura                       (todas las de estrategias\\Literatura)
    python -m scripts.backtest.backtest_literatura --reglas estrategias\\Literatura\\CL4_ReversionCierre1530_Stop06pct.toml
    python -m scripts.backtest.backtest_literatura --out-of-sample

Tipos de estrategia (campo `tipo` del archivo):
    ib_ruptura          ruptura del Initial Balance (Dalton)
    gap_and_go          gap confirmado por la primera vela (Plastun et al. 2020)
    asia_fade           falsa ruptura del rango asiatico en Londres
    reversion_cierre    reversion de la ultima media hora (Baltussen et al. 2021)
    delta_apertura      orderflow de la primera media hora (Chordia & Subrahmanyam 2004)

Convenciones (iguales a backtest.py): MNQ $2/pt, comision $1,50 ida y vuelta, 1 tick de slippage
al entrar y al salir, stop y otra salida en la misma vela = stop (peor caso), cuenta $3.000,
riesgo $300 por trade, minimo 1 contrato. Salida a mercado = cierre de la vela - 1 tick.

Salida: resultados\\Literatura\\in-sample\\<nombre>_trades.csv y comparacion.csv.
"""
import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.datos.construir_barras import leer_barras

RAIZ = Path(__file__).resolve().parents[2]
PROCESSED = RAIZ / "data" / "processed"
CARPETA = RAIZ / "estrategias" / "Literatura"
RESULTADOS = RAIZ / "resultados" / "Literatura"
PERIODOS = {"in-sample": (pd.Timestamp("2025-10-01"), pd.Timestamp("2026-03-31")),
            "out-of-sample": (pd.Timestamp("2026-04-01"), pd.Timestamp("2026-09-30"))}
CRITERIOS = RESULTADOS / "criterios_oos.md"
REGISTRO_OOS = RESULTADOS / "registro_oos.csv"  # cada corrida OOS queda anotada (una sola vez)
TICK = 0.25


# ---------------------------------------------------------------- datos

def cargar(inicio, fin):
    """Velas de 1 minuto desde el mes ANTERIOR a 'inicio' (para el cierre previo y la sesion
    asiatica del primer dia) hasta 'fin'; nunca lee meses posteriores. 'sesion' = dia de la
    sesion de futuros (desde las 18:00 pertenece al dia siguiente). 'ajuste' lleva precios de
    contratos viejos al ultimo (rolls)."""
    desde = f"{inicio - pd.DateOffset(months=1):%Y-%m}"
    archivos = [a for a in sorted(PROCESSED.glob("barras_1m_*.csv"))
                if desde <= a.stem[-7:] <= f"{fin:%Y-%m}"]
    v = pd.concat([leer_barras(a) for a in archivos], ignore_index=True)
    v = v.sort_values("ts_utc").reset_index(drop=True)
    hora = v["ts_ny"].dt.tz_localize(None)
    v["t"] = hora
    v["fecha"] = hora.dt.normalize()
    v["minuto"] = hora.dt.hour * 60 + hora.dt.minute
    v["sesion"] = (hora + pd.Timedelta(hours=6)).dt.normalize()
    v = v[v["sesion"] <= fin].reset_index(drop=True)
    roll = (v["symbol"] != v["symbol"].shift()).to_numpy(copy=True)
    roll[0] = False
    salto = np.where(roll, v["open"].to_numpy() - v["close"].shift().to_numpy(), 0.0)
    v["ajuste"] = np.cumsum(salto[::-1])[::-1] - salto
    return v


def por_dia(v, desde_min=None, hasta_min=None, por="fecha"):
    """{dia: arreglos} filtrado por minuto del dia (incluye desde, excluye hasta)."""
    out = {}
    for f, g in v.groupby(por, sort=True):
        if desde_min is not None:
            g = g[(g["minuto"] >= desde_min) & (g["minuto"] < hasta_min)]
        if len(g):
            out[f] = dict(o=g["open"].to_numpy(), h=g["high"].to_numpy(), l=g["low"].to_numpy(),
                          c=g["close"].to_numpy(), min=g["minuto"].to_numpy(), t=g["t"].to_numpy(),
                          vol=g["volumen"].to_numpy(float), delta=g["delta"].to_numpy(float),
                          sym=g["symbol"].to_numpy())
    return out


def minuto(texto):
    h, m = map(int, texto.split(":"))
    return h * 60 + m


def idx_min(dia, m):
    """indice de la vela que empieza en el minuto m (o la siguiente)."""
    k = np.searchsorted(dia["min"], m)
    return k if k < len(dia["min"]) else None


def idx_fin(dia, m):
    """indice de la ultima vela que empieza antes de m (su cierre = salida por tiempo)."""
    return np.searchsorted(dia["min"], m) - 1


class Datos:
    def __init__(self, inicio, fin):
        self.v = cargar(inicio, fin)
        self.rth = por_dia(self.v, 570, 960)
        self.ses = {s: x for s, x in por_dia(self.v, por="sesion").items() if s >= inicio}
        self.fechas = sorted(self.rth)
        # dias que se operan: dentro del periodo y completos (sin medios dias)
        self.completos = [f for f in self.fechas if f >= inicio and len(self.rth[f]["c"]) > 380]
        aj = self.v.groupby("fecha")["ajuste"].first()
        # cierre RTH previo llevado al contrato del dia (sin salto de roll)
        self.prev = {f: self.rth[p]["c"][-1] + aj[p] - aj[f]
                     for p, f in zip(self.fechas[:-1], self.fechas[1:])}


# ---------------------------------------------------------------- ejecucion de un trade

def simular(dia, i, d, stop, salida_idx, cf):
    """Entra a mercado en la apertura de la vela i (+1 tick), d = +1 largo / -1 corto.
    Sale por stop (toque; si abre mas alla, a la apertura) o al cierre de la vela salida_idx."""
    o, h, l, c = dia["o"], dia["h"], dia["l"], dia["c"]
    ent = o[i] + d * TICK
    riesgo = d * (ent - stop)
    if riesgo <= 0:
        return None
    n = max(int(cf["riesgo_usd"] // (riesgo * cf["valor_punto_usd"])), cf["minimo_contratos"])
    for j in range(i, salida_idx + 1):
        if (l[j] <= stop) if d > 0 else (h[j] >= stop):
            px = o[j] if (d > 0 and o[j] < stop) or (d < 0 and o[j] > stop) else stop
            salida, motivo, k = px - d * TICK, "stop", j
            break
        if j == salida_idx:
            salida, motivo, k = c[j] - d * TICK, "tiempo", j
            break
    pts = d * (salida - ent)
    return dict(entrada_hora=pd.Timestamp(dia["t"][i]), salida_hora=pd.Timestamp(dia["t"][k]),
                direccion="largo" if d > 0 else "corto", entrada=ent, stop=stop, salida=salida,
                motivo=motivo, riesgo_pts=riesgo, contratos=n,
                riesgo_usd=round(riesgo * cf["valor_punto_usd"] * n, 2), puntos=pts, R=round(pts / riesgo, 4),
                usd_neto=round(pts * cf["valor_punto_usd"] * n - cf["comision_ida_vuelta_usd"] * n, 2))


def redondear(x):
    return np.round(x / TICK) * TICK


# ---------------------------------------------------------------- estrategias

def ib_ruptura(D, r, cf):
    s = r["senal"]
    ib_fin, hasta = minuto(s["ib_fin"]), minuto(s["hasta"])
    tf, fin_m = s["vela_minutos"], minuto(r["salida"]["cierre_forzado"])
    trades = []
    for f in D.completos:
        dd = D.rth[f]
        if dd["min"][0] != 570:
            continue
        k0 = idx_min(dd, ib_fin)
        hi, lo = dd["h"][:k0].max(), dd["l"][:k0].min()
        fin = idx_fin(dd, fin_m)
        cierre_tf = (dd["min"] + 1 - 570) % tf == 0
        for j in range(k0, fin):
            if dd["min"][j] >= hasta:
                break
            if not cierre_tf[j]:
                continue
            lado = 1 if dd["c"][j] > hi else (-1 if dd["c"][j] < lo else 0)
            if lado:
                b = r["stop"]["buffer_ticks"] * TICK
                if r["stop"]["tipo"] == "lado_opuesto":
                    stp = lo - b if lado > 0 else hi + b
                else:  # "mitad"
                    stp = redondear((hi + lo) / 2)
                t = simular(dd, j + 1, lado, stp, fin, cf)
                if t:
                    trades.append(t)
                break  # solo la primera ruptura del dia
    return trades


def gap_and_go(D, r, cf):
    s = r["senal"]
    entrada, fin_m = minuto(s["entrada"]), minuto(r["salida"]["cierre_forzado"])
    trades = []
    for f in D.completos:
        if f not in D.prev:
            continue
        dd = D.rth[f]
        if dd["min"][0] != 570:
            continue
        pc = D.prev[f]
        gap = dd["o"][0] / pc - 1
        if not (s["gap_min_pct"] / 100 <= abs(gap) <= s["gap_max_pct"] / 100):
            continue
        d = 1 if gap > 0 else -1
        j = idx_min(dd, entrada) - 1
        if s["requiere_confirmacion"] and d * (dd["c"][j] - dd["o"][0]) <= 0:
            continue
        if r["stop"]["tipo"] == "cierre_previo":
            stp = redondear(pc)
        else:  # "extremo": extremo opuesto desde 09:30
            stp = dd["l"][:j + 1].min() - TICK if d > 0 else dd["h"][:j + 1].max() + TICK
        t = simular(dd, j + 1, d, stp, idx_fin(dd, fin_m), cf)
        if t:
            t["gap_pct"] = round(gap * 100, 4)
            trades.append(t)
    return trades


def asia_fade(D, r, cf):
    s = r["senal"]
    fin_asia, hasta = minuto(s["fin_rango_asia"]), minuto(s["hasta"])
    tf, salida = s["vela_minutos"], minuto(r["salida"]["cierre_forzado"])
    trades = []
    for ses in sorted(D.ses):
        x = D.ses[ses]
        m = x["min"]
        if len(set(x["sym"])) > 1:
            continue  # roll dentro de la sesion
        asia = (m >= 1080) | (m < fin_asia)
        if asia.sum() < 400:
            continue
        hi, lo = x["h"][asia].max(), x["l"][asia].min()
        orden = np.where(m >= 1080, m - 1440, m)  # minutos desde medianoche (noche negativa)
        idx = np.nonzero((orden >= fin_asia) & (orden < 570))[0]
        if len(idx) == 0:
            continue
        fin = idx[orden[idx] <= salida][-1]
        for j in idx:
            if orden[j] >= hasta:
                break
            if (m[j] + 1) % tf:
                continue
            lado = 1 if x["c"][j] > hi else (-1 if x["c"][j] < lo else 0)
            if lado:
                d = -lado if s["modo"] == "en_contra" else lado
                if s["modo"] == "en_contra":
                    stp = redondear(x["c"][j] + lado * (hi - lo) * r["stop"]["fraccion_rango"])
                else:
                    stp = lo - TICK if d > 0 else hi + TICK
                t = simular(x, j + 1, d, stp, fin, cf)
                if t:
                    t["asia_max"], t["asia_min"] = hi, lo
                    trades.append(t)
                break  # 1 trade por sesion
    return trades


def reversion_cierre(D, r, cf):
    hora = minuto(r["senal"]["hora_decision"])
    trades = []
    for f in D.completos:
        dd = D.rth[f]
        if dd["min"][0] != 570:
            continue
        j = idx_min(dd, hora) - 1
        d = -1 if dd["c"][j] > dd["o"][0] else 1
        stp = redondear(dd["c"][j] * (1 - d * r["stop"]["porcentaje"] / 100))
        t = simular(dd, j + 1, d, stp, len(dd["c"]) - 1, cf)  # salida al cierre de la vela 15:59
        if t:
            trades.append(t)
    return trades


def delta_apertura(D, r, cf):
    s = r["senal"]
    hora, fin_m = minuto(s["hora_decision"]), minuto(r["salida"]["cierre_forzado"])
    trades = []
    for f in D.completos:
        dd = D.rth[f]
        if dd["min"][0] != 570:
            continue
        j = idx_min(dd, hora) - 1
        x = dd["delta"][: j + 1].sum() / dd["vol"][: j + 1].sum()
        if abs(x) <= s["umbral"]:
            continue
        d = 1 if x > 0 else -1
        stp = redondear(dd["c"][j] * (1 - d * r["stop"]["porcentaje"] / 100))
        t = simular(dd, j + 1, d, stp, idx_fin(dd, fin_m), cf)
        if t:
            t["delta_normalizado"] = round(x, 5)
            trades.append(t)
    return trades


TIPOS = {"ib_ruptura": ib_ruptura, "gap_and_go": gap_and_go, "asia_fade": asia_fade,
         "reversion_cierre": reversion_cierre, "delta_apertura": delta_apertura}
COSTOS = {"valor_punto_usd": 2.0, "comision_ida_vuelta_usd": 1.5, "capital_inicial_usd": 3000.0,
          "riesgo_usd": 300.0, "minimo_contratos": 1}


# ---------------------------------------------------------------- resumen

def resumen(t, capital, corte):
    """corte = fecha que separa la 1a y la 2a mitad del periodo."""
    u = t["usd_neto"].to_numpy()
    saldo = capital + np.cumsum(u)
    pico = np.maximum.accumulate(np.r_[capital, saldo])[1:]
    gan, per = u[u > 0].sum(), -u[u < 0].sum()
    h1 = t["entrada_hora"] < corte
    return dict(trades=len(u), neto_usd=round(u.sum(), 2), saldo_final=round(saldo[-1], 2),
                profit_factor=round(gan / per, 2) if per else np.inf, ganadores_pct=round((u > 0).mean() * 100, 1),
                R_prom=round(t["R"].mean(), 3), max_drawdown_usd=round((saldo - pico).min(), 2),
                max_drawdown_pct=round(((saldo - pico) / pico).min() * 100, 1), saldo_minimo=round(saldo.min(), 2),
                mitad1_usd=round(u[h1].sum(), 2), mitad2_usd=round(u[~h1].sum(), 2),
                sin_top5_usd=round(np.sort(u)[::-1][5:].sum(), 2),
                largos_usd=round(u[t["direccion"] == "largo"].sum(), 2),
                cortos_usd=round(u[t["direccion"] == "corto"].sum(), 2))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reglas", nargs="+", help="archivos .toml (por defecto todos los de la carpeta)")
    p.add_argument("--out-of-sample", action="store_true",
                   help="abr-sep 2026: exige criterios_oos.md y cada archivo se corre UNA sola vez")
    p.add_argument("--repetir-oos", action="store_true", help="permite repetir un OOS ya corrido (no usar)")
    a = p.parse_args()
    periodo = "out-of-sample" if a.out_of_sample else "in-sample"
    archivos = [Path(x) for x in a.reglas] if a.reglas else \
        [x for x in sorted(CARPETA.glob("*.toml")) if not x.name.startswith("_")]
    if not archivos:
        sys.exit("No hay archivos .toml en estrategias\\Literatura.")
    reglas = {x: tomllib.loads(x.read_text(encoding="utf-8")) for x in archivos}
    for archivo, r in reglas.items():
        if r.get("tipo") not in TIPOS:
            sys.exit(f"{archivo.name}: tipo desconocido {r.get('tipo')!r}. Opciones: {', '.join(TIPOS)}")
    if a.out_of_sample:
        if not CRITERIOS.exists():
            sys.exit("Primero hay que escribir los criterios de aprobacion: " + str(CRITERIOS))
        ya = set(pd.read_csv(REGISTRO_OOS)["variante"]) if REGISTRO_OOS.exists() else set()
        repetidas = [r["nombre"] for r in reglas.values() if r["nombre"] in ya]
        if repetidas and not a.repetir_oos:
            sys.exit("El out-of-sample ya se corrio para: " + ", ".join(repetidas))
    inicio, fin = PERIODOS[periodo]
    print(f"Cargando velas {periodo} ({inicio:%Y-%m-%d} a {fin:%Y-%m-%d})...")
    D = Datos(inicio, fin)
    salida = RESULTADOS / periodo
    salida.mkdir(parents=True, exist_ok=True)
    filas = []
    for archivo, r in reglas.items():
        cf = {**COSTOS, **r.get("costos", {}), **r.get("riesgo", {})}
        t = pd.DataFrame(TIPOS[r["tipo"]](D, r, cf))
        t.to_csv(salida / f"{r['nombre']}_trades.csv", index=False)
        corte = inicio + pd.DateOffset(months=3)
        res = dict(variante=r["nombre"], archivo=archivo.name, tipo=r["tipo"], periodo=periodo,
                   **resumen(t, cf["capital_inicial_usd"], corte))
        filas.append(res)
        print(f"  {r['nombre']}: {res['trades']} trades, neto ${res['neto_usd']:,.2f}, PF {res['profit_factor']}")
    tabla = pd.DataFrame(filas)
    ruta = salida / "comparacion.csv"
    if ruta.exists():
        ant = pd.read_csv(ruta)
        activos = {x.name for x in CARPETA.glob("*.toml") if not x.name.startswith("_")}
        ant = ant[ant["archivo"].isin(activos) & ~ant["variante"].isin(tabla["variante"])]
        tabla = pd.concat([ant, tabla])
    tabla.sort_values("neto_usd", ascending=False).to_csv(ruta, index=False)
    if a.out_of_sample:
        log = pd.DataFrame(filas).assign(fecha_corrida=pd.Timestamp.now().strftime("%Y-%m-%d %H:%M"))
        if REGISTRO_OOS.exists():
            log = pd.concat([pd.read_csv(REGISTRO_OOS), log])
        log.to_csv(REGISTRO_OOS, index=False)
    print(f"Listo: {ruta}")


if __name__ == "__main__":
    main()
