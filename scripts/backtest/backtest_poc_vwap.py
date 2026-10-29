"""Backtest de la estrategia POC + VWAP (estrategias\\POC_VWAP). Todo en hora de Nueva York.

Uso:
    python -m scripts.backtest.backtest_poc_vwap                  (todas las de estrategias\\POC_VWAP)
    python -m scripts.backtest.backtest_poc_vwap --reglas estrategias\\POC_VWAP\\PV15_RTH_BE_Trail15M.toml
    python -m scripts.backtest.backtest_poc_vwap --reglas <archivos> --out-of-sample

Por defecto corre el in-sample (2025-10-01 a 2026-03-31). --out-of-sample (2026-04-01 a
2026-09-30) exige --reglas y corre cada variante una sola vez (repetir exige --repetir-oos),
con el mismo candado que scripts\\backtest\\backtest.py (registro.csv, huella de reglas).

Reglas (las cifras salen del archivo .toml de cada variante):
- Sesión de 18:00 (día anterior) a 16:00. POC en desarrollo y VWAP al cierre de cada vela
  de 1 minuto, calculados SOLO con los trades hasta ese cierre (regla inviolable:
  estrategias\\POC_VWAP\\REGLA_PERFIL_EN_DESARROLLO.md). Datos: perfil_vwap.minutos_del_mes.
- Caja: existe cuando |POC - VWAP| <= caja.tolerancia_puntos; va del menor de los dos menos
  caja.margen_puntos al mayor más caja.margen_puntos (el margen agranda la zona, no la
  tolerancia). Toques, salidas y stop se miden contra esta zona.
- Lado: el del último cierre de 1 minuto fuera de una caja existente (arriba / abajo).
- Intento: una vela toca la caja (basta la mecha) y ya se sabe el lado -> se abre un intento
  "entró por arriba/abajo". Si la caja nace con el precio adentro no hay lado: no se opera
  hasta que salga y vuelva a entrar. Si la caja desaparece, el intento se cancela.
- Señal: la primera vela que cierra >= caja.cierre_fuera_ticks fuera de la caja resuelve el
  intento: mismo lado por el que entró = rechazo; lado contrario = ruptura (una vela que cruza
  la caja entera también es ruptura). Cierra arriba -> largo, abajo -> corto.
  Si hay un trade abierto, la señal se ignora y el intento termina.
- Entrada a mercado en la apertura del minuto siguiente (+ slippage). Un trade a la vez.
- Stop: lado opuesto de la caja al momento de la señal +/- stop.buffer_puntos. Queda fijo;
  solo lo mueven el breakeven y el trailing.
- Al tocar 1R (gestion.activar_en_R): con breakeven, stop a la entrada. Desde ahí trailing:
  stop 1 tick bajo el mínimo (largos) / sobre el máximo (cortos) de la última vela de
  gestion.trailing_minutos YA CERRADA (alineadas al reloj), solo si reduce el riesgo.
  Rige desde la vela siguiente.
- Objetivo (gestion.objetivo_R, 0 = sin objetivo): orden límite en entrada + objetivo_R x riesgo;
  se llena si el precio pasa ejecucion.limite_requiere_traspaso_ticks más allá (al objetivo, o
  a la apertura si abre más allá), sin slippage.
- Si en una vela de 1 minuto se tocan el stop y el 1R (o el objetivo), se supone que primero
  fue el stop.
- Ventanas: nocturno 20:00-09:00 (cierre forzado a las 09:00) y RTH 09:30-15:55 (cierre
  forzado a las 15:55). Las señales cuentan solo dentro de la ventana; nunca pasa un trade
  de una ventana a otra.
- Límite (horario.max_trades_por_ventana, 0 = sin límite): máximo de entradas ejecutadas por
  ventana y sesión (los giros cuentan). Alcanzado el límite, las señales de esa ventana se ignoran.
"""
import argparse
import sys
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.backtest import (PERIODOS, REGLAS, TICK, anotar, carpeta_resultados,
                                       combinar, guardar_reglas, huella, huella_reglas, leer_registro,
                                       redondear, reglas_ya_en_oos, resumir)
from scripts.indicadores.perfil_vwap import minutos_del_mes
from scripts.datos.validar import CalendarioNYSE

ESTRATEGIA = "POC_VWAP"
INICIO_SESION = 18 * 60  # los minutos se cuentan desde las 18:00 del día anterior
DEFAULTS = {
    "nombre": "",
    "descripcion": "",
    "horario": {"ventanas": ["nocturno", "rth"],
                "nocturno_inicio": "20:00", "nocturno_fin": "09:00",
                "rth_inicio": "09:30", "rth_fin": "15:55", "max_trades_por_ventana": 0},
    "caja": {"tolerancia_puntos": 15.0, "margen_puntos": 0.0, "cierre_fuera_ticks": 5,
             "tipos": ["rechazo", "ruptura"]},
    "stop": {"buffer_puntos": 5.0},
    "gestion": {"activar_en_R": 1.0, "breakeven": True, "trailing_minutos": 15,
                "trailing_buffer_ticks": 1, "objetivo_R": 0.0},
    "filtros": {"excluir_feriados": True, "excluir_dias_reducidos": True},
    "ejecucion": {"slippage_entrada_ticks": 1, "slippage_salida_ticks": 1,
                  "limite_requiere_traspaso_ticks": 1},
    "costos": {"valor_punto_usd": 2, "comision_ida_vuelta_usd": 1.5},
    "riesgo": {"capital_inicial_usd": 3000, "riesgo_usd": 300, "minimo_contratos": 1},
}
COLUMNAS_TRADES = [
    "fecha", "contrato", "ventana", "tipo", "direccion", "hora_senal", "hora_entrada",
    "precio_entrada", "caja_alta", "caja_baja", "poc", "vwap", "stop_inicial", "objetivo", "hora_salida",
    "precio_salida", "motivo_salida", "puntos", "R", "contratos", "riesgo_usd", "usd_bruto",
    "usd_neto", "mfe_R", "mae_R", "duracion_min", "stops",
]
# parámetros agregados después: no cambian la huella mientras tengan su valor por defecto
NUEVOS = {("caja", "margen_puntos"): 0.0, ("gestion", "objetivo_R"): 0.0,
          ("ejecucion", "limite_requiere_traspaso_ticks"): 1, ("horario", "max_trades_por_ventana"): 0}


def sin_nuevos(r):
    """Reglas sin los parámetros NUEVOS que están en su valor por defecto (para la huella)."""
    r = {k: dict(v) if isinstance(v, dict) else v for k, v in r.items()}
    for (seccion, clave), defecto in NUEVOS.items():
        if r[seccion].get(clave) == defecto:
            del r[seccion][clave]
    return r


# ---------------------------------------------------------------- reglas

def minuto_sesion(texto, archivo):
    """'HH:MM' -> minutos desde las 18:00 del día anterior (20:00 -> 120, 09:30 -> 930)."""
    try:
        hh, mm = map(int, texto.split(":"))
    except ValueError:
        raise SystemExit(f"{archivo.name}: la hora '{texto}' debe tener el formato HH:MM.")
    return (hh * 60 + mm - INICIO_SESION) % (24 * 60)


def cargar_reglas(archivo):
    try:
        crudo = tomllib.loads(archivo.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"{archivo.name}: el archivo tiene un error de formato ({e}).")
    r = combinar(DEFAULTS, crudo, archivo)
    r["nombre"] = r["nombre"] or archivo.stem
    h, c, g = r["horario"], r["caja"], r["gestion"]
    if not set(h["ventanas"]) <= {"nocturno", "rth"} or not h["ventanas"]:
        raise SystemExit(f"{archivo.name}: horario.ventanas debe usar 'nocturno' y/o 'rth'.")
    if not set(c["tipos"]) <= {"rechazo", "ruptura"} or not c["tipos"]:
        raise SystemExit(f"{archivo.name}: caja.tipos debe usar 'rechazo' y/o 'ruptura'.")
    if g["trailing_minutos"] not in (1, 5, 15, 30, 60):
        raise SystemExit(f"{archivo.name}: gestion.trailing_minutos debe ser 1, 5, 15, 30 o 60.")
    ventanas = {}
    for v in h["ventanas"]:
        ini, fin = minuto_sesion(h[f"{v}_inicio"], archivo), minuto_sesion(h[f"{v}_fin"], archivo)
        if not 0 <= ini < fin <= 22 * 60:  # 22 h = hasta las 16:00
            raise SystemExit(f"{archivo.name}: la ventana {v} debe quedar entre 18:00 y 16:00.")
        ventanas[v] = (ini, fin)
    return r, ventanas


# ---------------------------------------------------------------- datos

def cargar_minutos(periodo):
    """Tabla minuto a minuto (POC/VWAP en desarrollo + velas) de las sesiones del periodo."""
    desde, hasta = map(pd.Timestamp, PERIODOS[periodo])
    meses = pd.period_range(desde, hasta, freq="M").strftime("%Y-%m")
    t = pd.concat([minutos_del_mes(m) for m in meses], ignore_index=True)
    t = t[(t["sesion"] >= desde) & (t["sesion"] <= hasta)].sort_values("ts_utc")
    hora = t["ts_utc"].dt.tz_convert("America/New_York").dt.tz_localize(None)
    t["min_ses"] = ((hora - (t["sesion"] - pd.Timedelta(days=1)))
                    / pd.Timedelta(minutes=1)).astype(int) - INICIO_SESION
    t["hora_ny"] = hora
    return t.reset_index(drop=True)


def sesiones_validas(t):
    """{sesión: arreglos} sin feriados ni días de horario reducido (sin velas desde las 13:30)."""
    feriados = set(CalendarioNYSE().holidays(t["sesion"].min(), t["sesion"].max()))
    salida = {}
    for dia, s in t.groupby("sesion", sort=True):
        reducido = s["min_ses"].max() < 13 * 60 + 30 + 6 * 60  # 13:30 en minutos de sesión
        salida[dia] = dict(
            feriado=dia in feriados, reducido=reducido,
            **{k: s[k].to_numpy() for k in ("min_ses", "open", "high", "low", "close", "poc",
                                            "vwap", "symbol")},
            hora=s["hora_ny"].to_numpy())
    return salida


# ---------------------------------------------------------------- simulación

def bloques_cerrados(m, h, l, minutos):
    """Para cada vela de 1 minuto: máximo y mínimo de la última vela de 'minutos' ya cerrada
    al CIERRE de esa vela de 1 minuto (alineadas al reloj, desde las 18:00). NaN si ninguna."""
    bloque = m // minutos
    ids, inicio = np.unique(bloque, return_index=True)
    alto = np.maximum.reduceat(h, inicio)
    bajo = np.minimum.reduceat(l, inicio)
    fin = (ids + 1) * minutos  # minuto (de sesión) en que termina cada bloque
    k = np.searchsorted(fin, m + 1, side="right") - 1  # último bloque con fin <= cierre de la vela
    ok = k >= 0
    return (np.where(ok, alto[np.maximum(k, 0)], np.nan),
            np.where(ok, bajo[np.maximum(k, 0)], np.nan))


def simular_sesion(r, ventanas, fecha, s):
    m, o, h, l, c = s["min_ses"], s["open"], s["high"], s["low"], s["close"]
    poc, vwap = s["poc"], s["vwap"]
    cj, st, g, ej, co, rs = (r[k] for k in ("caja", "stop", "gestion", "ejecucion", "costos", "riesgo"))
    tol, fuera = cj["tolerancia_puntos"], cj["cierre_fuera_ticks"] * TICK
    slip_e, slip_s = ej["slippage_entrada_ticks"] * TICK, ej["slippage_salida_ticks"] * TICK
    tr_buffer = g["trailing_buffer_ticks"] * TICK
    traspaso = ej["limite_requiere_traspaso_ticks"] * TICK
    tr_alto, tr_bajo = bloques_cerrados(m, h, l, g["trailing_minutos"])
    existe = np.abs(poc - vwap) <= tol
    tope = np.maximum(poc, vwap) + cj["margen_puntos"]
    piso = np.minimum(poc, vwap) - cj["margen_puntos"]
    # la caja y el intento se siguen desde las 20:00 en todas las variantes (así las señales de
    # RTH son las mismas aunque la variante no opere de noche)
    inicio_estado = minuto_sesion(r["horario"]["nocturno_inicio"], Path(r["nombre"]))

    def ventana_de(t):
        for nombre, (ini, fin) in ventanas.items():
            if ini <= t < fin:
                return nombre
        return None

    trades, pos, pendiente = [], None, None
    lado, intento = 0, 0  # lado: +1 arriba / -1 abajo / 0 sin lado; intento: lado por donde entró
    tope_trades = r["horario"]["max_trades_por_ventana"]
    entradas = dict.fromkeys(ventanas, 0)  # entradas ejecutadas por ventana en esta sesión

    def cerrar(precio, i, motivo):
        nonlocal pos
        p = pos
        d, riesgo = p["d"], p["riesgo"]
        puntos = d * (precio - p["entrada"])
        p["mfe"], p["mae"] = max(p["mfe"], puntos), min(p["mae"], puntos)
        bruto = puntos * co["valor_punto_usd"] * p["n"]
        trades.append(dict(
            fecha=fecha.date().isoformat(), contrato=s["symbol"][p["i"]], ventana=p["ventana"],
            tipo=p["tipo"], direccion="largo" if d > 0 else "corto",
            hora_senal=f"{pd.Timestamp(s['hora'][p['i_senal']]):%Y-%m-%d %H:%M}",
            hora_entrada=f"{pd.Timestamp(s['hora'][p['i']]):%Y-%m-%d %H:%M}",
            precio_entrada=p["entrada"], caja_alta=round(p["tope"], 4), caja_baja=round(p["piso"], 4),
            poc=p["poc"], vwap=round(p["vwap"], 4), stop_inicial=p["stop_ini"],
            objetivo="" if p["objetivo"] is None else p["objetivo"],
            hora_salida=f"{pd.Timestamp(s['hora'][i]):%Y-%m-%d %H:%M}", precio_salida=precio,
            motivo_salida=motivo, puntos=round(puntos, 4), R=round(puntos / riesgo, 3),
            contratos=p["n"], riesgo_usd=round(riesgo * co["valor_punto_usd"] * p["n"], 2),
            usd_bruto=round(bruto, 2),
            usd_neto=round(bruto - co["comision_ida_vuelta_usd"] * p["n"], 2),
            mfe_R=round(p["mfe"] / riesgo, 3), mae_R=round(p["mae"] / riesgo, 3),
            duracion_min=int(m[i] - m[p["i"]]),
            stops=";".join(f"{pd.Timestamp(s['hora'][k]):%Y-%m-%d %H:%M}={x}" for k, x in p["stops"])))
        pos = None

    def mover(nuevo, motivo, i):
        """Mueve el stop (rige desde la vela siguiente) y anota el cambio."""
        pos["stop"], pos["motivo"] = nuevo, motivo
        pos["stops"].append((min(i + 1, len(m) - 1), nuevo))

    for i in range(len(m)):
        t = m[i]
        # 1) cierre forzado al fin de la ventana del trade (en la apertura de esta vela)
        if pos and t >= pos["cierre"]:
            cerrar(o[i] - pos["d"] * slip_s, i, "hora_cierre")
        # 2) entrada pendiente: a mercado en la apertura, si sigue dentro de su ventana
        recien = False
        if pendiente and not pos:
            d, k, nombre = pendiente
            ini, fin = ventanas[nombre]
            if t < fin:
                entrada = o[i] + d * slip_e
                stop = redondear((piso[k] - st["buffer_puntos"]) if d > 0 else (tope[k] + st["buffer_puntos"]))
                riesgo = d * (entrada - stop)
                if riesgo > 0:  # si abrió más allá del stop, no se entra
                    n = max(int(rs["riesgo_usd"] // (riesgo * co["valor_punto_usd"])),
                            rs["minimo_contratos"])
                    if n >= 1:
                        pos = dict(d=d, entrada=entrada, stop=stop, stop_ini=stop, riesgo=riesgo,
                                   n=n, i=i, i_senal=k, ventana=nombre, cierre=fin,
                                   tipo=pendiente_tipo, tope=tope[k], piso=piso[k], poc=poc[k],
                                   vwap=vwap[k], mfe=0.0, mae=0.0, activado=False,
                                   motivo="stop", stops=[(i, stop)],
                                   objetivo=redondear(entrada + d * g["objetivo_R"] * riesgo)
                                   if g["objetivo_R"] else None)
                        recien = True
                        entradas[nombre] += 1
        pendiente = None
        # 3) stop dentro de la vela (se revisa antes que el 1R: "lo peor")
        if pos:
            d = pos["d"]
            desde = pos["entrada"] if recien else o[i]
            if min(d * h[i], d * l[i]) <= d * pos["stop"]:
                cerrar(d * (min(d * desde, d * pos["stop"])) - d * slip_s, i, pos["motivo"])
            elif pos["objetivo"] is not None and max(d * h[i], d * l[i]) >= d * pos["objetivo"] + traspaso:
                cerrar(d * max(d * desde, d * pos["objetivo"]), i, "objetivo")
        # 4) al cierre de la vela: MFE/MAE, 1R -> breakeven + trailing (rigen desde la siguiente)
        if pos:
            d, entrada = pos["d"], pos["d"] * pos["entrada"]
            pos["mfe"] = max(pos["mfe"], max(d * h[i], d * l[i]) - entrada)
            pos["mae"] = min(pos["mae"], min(d * h[i], d * l[i]) - entrada)
            if not pos["activado"] and pos["mfe"] >= g["activar_en_R"] * pos["riesgo"]:
                pos["activado"] = True
                if g["breakeven"] and entrada > d * pos["stop"]:
                    mover(pos["entrada"], "breakeven", i)
            if pos["activado"]:
                extremo = tr_bajo[i] if d > 0 else tr_alto[i]
                if not np.isnan(extremo):
                    nuevo = redondear(extremo - d * tr_buffer)
                    if d * nuevo > d * pos["stop"]:
                        mover(nuevo, "trailing", i)
        # 5) caja e intento al cierre de la vela
        if t < inicio_estado:
            continue
        if not existe[i]:
            lado, intento = 0, 0
            continue
        if intento == 0 and lado and l[i] <= tope[i] and h[i] >= piso[i]:
            intento = lado
        if intento:
            salida = 1 if c[i] >= tope[i] + fuera else -1 if c[i] <= piso[i] - fuera else 0
            if salida:
                tipo = "rechazo" if salida == intento else "ruptura"
                nombre = ventana_de(t)
                libre = nombre and (not tope_trades or entradas[nombre] < tope_trades)
                if not pos and libre and tipo in cj["tipos"]:
                    pendiente, pendiente_tipo = (salida, i, nombre), tipo
                intento = 0
        lado = 1 if c[i] > tope[i] else -1 if c[i] < piso[i] else lado
    if pos:  # la sesión terminó antes de la hora de cierre
        cerrar(c[-1], len(m) - 1, "fin_de_datos")
    return trades


def correr(r, ventanas, sesiones):
    fl = r["filtros"]
    trades, evaluadas = [], []
    for fecha, s in sesiones.items():
        if (fl["excluir_feriados"] and s["feriado"]) or (fl["excluir_dias_reducidos"] and s["reducido"]):
            continue
        evaluadas.append(fecha)
        trades += simular_sesion(r, ventanas, fecha, s)
    return pd.DataFrame(trades, columns=COLUMNAS_TRADES), evaluadas


def main():
    parser = argparse.ArgumentParser(description="Backtest POC + VWAP.")
    parser.add_argument("--reglas", nargs="+", help=f"Archivos .toml (por defecto: estrategias\\{ESTRATEGIA})")
    parser.add_argument("--out-of-sample", action="store_true",
                        help="Corre 2026-04-01 a 2026-09-30 (una vez por variante; exige --reglas)")
    parser.add_argument("--repetir-oos", action="store_true",
                        help="Permite repetir out-of-sample con una variante ya probada ahí")
    args = parser.parse_args()
    if args.repetir_oos and not args.out_of_sample:
        sys.exit("--repetir-oos solo tiene sentido junto con --out-of-sample.")
    if args.out_of_sample and not args.reglas:
        sys.exit("El out-of-sample se corre solo con las variantes elegidas: indica --reglas.")
    archivos = ([Path(a) for a in args.reglas] if args.reglas else
                [a for a in sorted((REGLAS / ESTRATEGIA).glob("*.toml")) if not a.name.startswith("_")])
    if not archivos:
        sys.exit(f"No hay archivos .toml en estrategias\\{ESTRATEGIA}.")
    variantes = [(a, *cargar_reglas(a)) for a in archivos]
    nombres = [r["nombre"] for _, r, _ in variantes]
    if len(set(nombres)) < len(nombres):
        sys.exit("Hay variantes con el mismo nombre.")

    periodo = "out-of-sample" if args.out_of_sample else "in-sample"
    if args.out_of_sample:
        print("=" * 70)
        print(" OUT-OF-SAMPLE (2026-04-01 a 2026-09-30): la prueba final. Úsala UNA vez,")
        print(" con las variantes ya elegidas y el criterio de aprobación ya decidido.")
        print("=" * 70)
        ya = reglas_ya_en_oos(leer_registro())
        for a, r, _ in variantes:
            if huella_reglas(sin_nuevos(r)) in ya and not args.repetir_oos:
                sys.exit(f"Las reglas de '{r['nombre']}' ({a.name}) ya se probaron en out-of-sample. "
                         "Repetir le quita valor a la prueba; si de verdad hace falta, agrega --repetir-oos.")
    print(f"Periodo {periodo}: {PERIODOS[periodo][0]} a {PERIODOS[periodo][1]} (sesiones NY)")
    sesiones = sesiones_validas(cargar_minutos(periodo))
    carpeta = carpeta_resultados(ESTRATEGIA, periodo)
    carpeta.mkdir(parents=True, exist_ok=True)
    filas = []
    for a, r, ventanas in variantes:
        trades, evaluadas = correr(r, ventanas, sesiones)
        res = resumir(trades, evaluadas, r["riesgo"]["capital_inicial_usd"])
        trades.to_csv(carpeta / f"{r['nombre']}_trades.csv", index=False)
        base = sin_nuevos(r)
        anotar(periodo, r["nombre"], a, huella(base), huella_reglas(base), res)
        guardar_reglas(a, base)
        filas.append({"variante": r["nombre"], "archivo": a.name, "huella": huella(base), **res})
        print(f"  {r['nombre']}: {res['trades']} trades, neto ${res['neto_usd']:,.2f}")
    tabla = pd.DataFrame(filas).sort_values("saldo_final_usd", ascending=False)
    tabla.to_csv(carpeta / "comparacion.csv", index=False)
    print(tabla[["variante", "trades", "pct_acierto", "profit_factor", "neto_usd",
                 "max_drawdown_pct", "saldo_final_usd", "saldo_minimo_usd"]].to_string(index=False))
    print(f"Trades en resultados\\{ESTRATEGIA}\\{periodo}\\<variante>_trades.csv")


if __name__ == "__main__":
    main()
