"""Backtest de estrategias intradía sobre NQ (ORB y otras). Todo en hora de Nueva York.

Uso:
    python -m scripts.backtest.backtest --estrategia ORB   (todas las de estrategias\\ORB)
    python -m scripts.backtest.backtest                              (todas las de todas las subcarpetas)
    python -m scripts.backtest.backtest --reglas estrategias\\ORB\\mi_orb.toml estrategias\\ORB\\otra.toml
    python -m scripts.backtest.backtest --reglas estrategias\\ORB\\mi_orb.toml --out-of-sample
    python -m scripts.backtest.backtest --reglas estrategias\\ORB\\mi_orb.toml --completo

La carpeta reglas tiene una subcarpeta por estrategia (ej. ORB, ORB).
Cada archivo .toml de una subcarpeta es una variante (los que empiezan con "_" se ignoran).
Los parámetros que un archivo no menciona usan su valor por defecto, que es el que aparece
en estrategias\\ORB\\_plantilla_base.toml (donde están explicados).

Periodos (por fecha de Nueva York):
- in-sample (por defecto): 2025-10-01 a 2026-03-31. Para probar y comparar variantes.
- out-of-sample (BLOQUEADO): 2026-04-01 a 2026-09-30. Solo con --out-of-sample y cada
  variante una sola vez (repetir exige --repetir-oos). Una corrida in-sample ni siquiera
  lee los archivos de esos meses.
- completo: 2025-10-01 a 2026-09-30 (los 12 meses). Solo con --completo y solo para
  variantes que ya se probaron en out-of-sample (así no sirve para mirarlo antes).

Simulación con velas de 1 minuto. Las señales usan velas de "timeframe_minutos" armadas
desde las de 1 minuto, y lo que se decide al cierre de una vela se ejecuta en la apertura
de la vela siguiente (nunca se mira el futuro). Si en una misma vela de 1 minuto se tocan
el stop y el objetivo, misma_vela = "peor" asume que primero se tocó el stop.

Resultados en la carpeta resultados: en resultados\\<estrategia>\\<periodo> los trades de
cada variante y la tabla comparativa; en resultados\\registro.csv todas las corridas y en
resultados\\reglas_probadas una copia de las reglas de cada variante corrida (para
compararlas: scripts\\informes\\comparar.py). registro.csv es uno solo para todas las estrategias:
así el candado del out-of-sample funciona aunque se mueva un archivo de carpeta.
"""
import argparse
import csv
import hashlib
import json
import math
import re
import sys
import tomllib
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.datos.construir_barras import PROCESSED, RAIZ, leer_barras
from scripts.datos.validar import CalendarioNYSE

REGLAS = RAIZ / "estrategias"
RESULTADOS = RAIZ / "resultados"
REGISTRO = RESULTADOS / "registro.csv"
REGLAS_PROBADAS = RESULTADOS / "reglas_probadas"
# estrategias con su propio motor: backtest.py no corre sus archivos
OTROS_MOTORES = {"POC_VWAP": "python -m scripts.backtest.backtest_poc_vwap",
                 "Literatura": "python -m scripts.backtest.backtest_literatura"}
PERIODOS = {
    "in-sample": ("2025-10-01", "2026-03-31"),
    "out-of-sample": ("2026-04-01", "2026-09-30"),
    "completo": ("2025-10-01", "2026-09-30"),  # incluye out-of-sample: solo tras probarlo ahí
}
TICK = 0.25
DIAS = ["lun", "mar", "mie", "jue", "vie", "sab", "dom"]
INICIO_RTH, FIN_RTH = 9 * 60 + 30, 16 * 60
TARDE = 13 * 60 + 30  # un día sin velas desde las 13:30 es un día de horario reducido

# Valores por defecto (iguales a estrategias\ORB\_plantilla_base.toml salvo las líneas
# marcadas "<<< BASE", donde están explicados).
DEFAULTS = {
    "nombre": "",
    "descripcion": "",
    "rango": {
        "inicio": "09:30", "minutos": 15,
        "ancho_min_puntos": 0, "ancho_max_puntos": 0,
        "ancho_min_vs_promedio": 0, "ancho_max_vs_promedio": 0, "dias_promedio": 14,
    },
    "entrada": {
        "timeframe_minutos": 5, "tipo": "breakout", "confirmacion": "cierre",
        "buffer_ticks": 1, "direccion": "ambas", "hora_limite": "11:00",
        "max_trades_dia": 1, "permitir_misma_direccion": True, "velas_para_ruptura": 0,
    },
    "retesteo": {"offset_ticks": 0, "max_velas_espera": 6, "cancelar_si_cierra_dentro": False,
                 "requiere_ruptura_previa": True, "ruptura_previa_en": "minuto"},
    "stop": {
        "tipo": "rango_opuesto", "valor": 0, "buffer_ticks": 1,
        "max_puntos": 0, "si_excede_max": "saltar", "respaldo": "rango_opuesto",
        "min_puntos": 0, "si_menor_min": "saltar",
    },
    "ema": {"periodos": 20, "timeframe_minutos": 5},
    "breakeven": {"activar_en_R": 0, "offset_ticks": 1},
    "trailing": {"tipo": "ninguno", "activar_en_R": 0, "distancia": 0, "velas": 2,
                 "timeframe_minutos": 0},
    "salida": {
        "objetivo_tipo": "R", "objetivo_valor": 2, "hora_cierre": "15:55",
        "salir_si_cierra_dentro": False,
    },
    "parcial": {"porcentaje": 0, "en_R": 1, "breakeven_tras_parcial": True},
    "filtros": {
        "dias_semana": ["lun", "mar", "mie", "jue", "vie"], "excluir_feriados": True,
        "excluir_dias_reducidos": True, "excluir_fechas": [],
        "gap_min_puntos": 0, "gap_max_puntos": 0, "a_favor_de": "ninguno",
    },
    "ejecucion": {
        "slippage_entrada_ticks": 1, "slippage_salida_ticks": 1,
        "limite_requiere_traspaso_ticks": 1, "misma_vela": "peor",
    },
    "costos": {"valor_punto_usd": 2, "contratos": 1, "comision_ida_vuelta_usd": 1.5},
    "riesgo": {"capital_inicial_usd": 3000, "modo": "riesgo_usd", "riesgo_usd": 300,
               "minimo_contratos": 1},
}
OPCIONES = {
    ("entrada", "tipo"): {"breakout", "retesteo", "retesteo_cierre", "cierre_ema"},
    ("entrada", "confirmacion"): {"toque", "cierre"},
    ("entrada", "direccion"): {"ambas", "solo_largos", "solo_cortos"},
    ("stop", "tipo"): {"rango_opuesto", "medio_rango", "puntos", "multiplo_rango", "vela_senal",
                       "ema"},
    ("stop", "si_excede_max"): {"saltar", "recortar"},
    ("stop", "si_menor_min"): {"saltar", "ampliar"},
    ("retesteo", "ruptura_previa_en"): {"minuto", "vela"},
    ("stop", "respaldo"): {"rango_opuesto", "medio_rango", "puntos", "multiplo_rango",
                           "vela_senal", "no_operar"},
    ("trailing", "tipo"): {"ninguno", "puntos", "R", "velas", "ema"},
    ("salida", "objetivo_tipo"): {"R", "puntos", "multiplo_rango", "ninguno"},
    ("filtros", "a_favor_de"): {"ninguno", "vwap", "gap"},
    ("ejecucion", "misma_vela"): {"peor", "mejor"},
    ("riesgo", "modo"): {"riesgo_usd", "contratos_fijos"},
}
COLUMNAS_TRADES = [
    "fecha", "contrato", "direccion", "hora_entrada", "precio_entrada", "stop_inicial",
    "stop_usado", "objetivo", "hora_salida", "precio_salida", "motivo_salida", "puntos", "R",
    "contratos", "riesgo_usd", "usd_bruto", "usd_neto", "rango_alto", "rango_bajo",
    "ancho_rango", "mfe_R", "mae_R", "duracion_min",
]
COLUMNAS_RESUMEN = [
    "trades", "dias_evaluados", "dias_con_trade", "pct_acierto", "neto_usd", "promedio_usd",
    "ganancia_prom_usd", "perdida_prom_usd", "profit_factor", "expectativa_R",
    "max_drawdown_usd", "racha_perdedora", "sharpe_anual", "largos", "neto_largos_usd",
    "cortos", "neto_cortos_usd", "capital_inicial_usd", "saldo_final_usd", "retorno_pct",
    "max_drawdown_pct", "saldo_minimo_usd", "riesgo_prom_usd", "riesgo_max_usd",
]


# ---------------------------------------------------------------- reglas

def combinar(base, nuevo, archivo, ruta=""):
    """Pone los valores del archivo sobre los de por defecto. Avisa de nombres mal escritos
    y de valores del tipo equivocado."""
    if not isinstance(nuevo, dict):
        raise SystemExit(f"{archivo.name}: [{ruta.rstrip('.')}] debe ser una sección.")
    for clave in nuevo:
        if clave not in base:
            raise SystemExit(f"{archivo.name}: parámetro desconocido '{ruta}{clave}' "
                             "(¿error de tipeo?).")
    salida = {}
    for clave, defecto in base.items():
        if isinstance(defecto, dict):
            salida[clave] = combinar(defecto, nuevo.get(clave, {}), archivo, f"{ruta}{clave}.")
            continue
        valor = nuevo.get(clave, defecto)
        es_numero = isinstance(defecto, (int, float)) and not isinstance(defecto, bool)
        ok = (isinstance(valor, bool) if isinstance(defecto, bool)
              else isinstance(valor, (int, float)) and not isinstance(valor, bool) if es_numero
              else isinstance(valor, type(defecto)))
        if not ok:
            raise SystemExit(f"{archivo.name}: '{ruta}{clave}' tiene un valor del tipo "
                             f"equivocado ({valor!r}); ejemplo válido: {defecto!r}.")
        salida[clave] = valor
    return salida


def a_minutos(texto, archivo, nombre):
    if not re.fullmatch(r"\d{2}:\d{2}", texto):
        raise SystemExit(f"{archivo.name}: {nombre} debe tener el formato HH:MM (ej. 09:30).")
    hh, mm = map(int, texto.split(":"))
    return hh * 60 + mm


def cargar_reglas(archivo):
    """Lee y valida un archivo de reglas. Devuelve (reglas completas, horas en minutos)."""
    try:
        crudo = tomllib.loads(archivo.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as e:
        raise SystemExit(f"{archivo.name}: el archivo tiene un error de formato ({e}).")
    r = combinar(DEFAULTS, crudo, archivo)
    r["nombre"] = r["nombre"] or archivo.stem

    def error(texto):
        raise SystemExit(f"{archivo.name}: {texto}")

    if not re.fullmatch(r"[A-Za-z0-9_\-]+", r["nombre"]):
        error("nombre solo puede tener letras sin tilde, números, '-' y '_'.")
    for (seccion, clave), validas in OPCIONES.items():
        if r[seccion][clave] not in validas:
            error(f"{seccion}.{clave} = {r[seccion][clave]!r} no es válido. "
                  f"Opciones: {', '.join(sorted(validas))}.")
    for seccion, valores in r.items():
        if isinstance(valores, dict):
            for clave, v in valores.items():
                if isinstance(v, (int, float)) and not isinstance(v, bool) and v < 0:
                    error(f"{seccion}.{clave} no puede ser negativo.")

    horas = {
        "inicio": a_minutos(r["rango"]["inicio"], archivo, "rango.inicio"),
        "hora_limite": a_minutos(r["entrada"]["hora_limite"], archivo, "entrada.hora_limite"),
        "hora_cierre": a_minutos(r["salida"]["hora_cierre"], archivo, "salida.hora_cierre"),
    }
    e, rg = r["entrada"], r["rango"]
    for seccion, clave in [("rango", "minutos"), ("entrada", "timeframe_minutos"),
                           ("entrada", "max_trades_dia"), ("rango", "dias_promedio"),
                           ("retesteo", "max_velas_espera"), ("trailing", "velas"),
                           ("costos", "contratos"), ("ema", "periodos"),
                           ("ema", "timeframe_minutos")]:
        v = r[seccion][clave]
        if not isinstance(v, int) or v < 1:
            error(f"{seccion}.{clave} debe ser un número entero de 1 o más.")
    if not isinstance(r["trailing"]["timeframe_minutos"], int):
        error("trailing.timeframe_minutos debe ser un número entero (0 = el de la entrada).")
    if e["tipo"] == "cierre_ema" and rg["minutos"] != r["ema"]["timeframe_minutos"]:
        error("con entrada.tipo = 'cierre_ema', rango.minutos debe ser igual a "
              "ema.timeframe_minutos (la vela de señal es una vela de la EMA).")
    rs = r["riesgo"]
    if not isinstance(rs["minimo_contratos"], int):
        error("riesgo.minimo_contratos debe ser un número entero (0 o más).")
    if rs["modo"] == "riesgo_usd" and not rs["riesgo_usd"] > 0:
        error("riesgo.riesgo_usd debe ser mayor que 0 con riesgo.modo = 'riesgo_usd'.")
    if not r["costos"]["valor_punto_usd"] > 0 or not rs["capital_inicial_usd"] > 0:
        error("costos.valor_punto_usd y riesgo.capital_inicial_usd deben ser mayores que 0.")
    if not isinstance(e["velas_para_ruptura"], int):
        error("entrada.velas_para_ruptura debe ser un número entero (0 = sin límite).")
    if rg["minutos"] % e["timeframe_minutos"]:
        error("rango.minutos debe ser múltiplo de entrada.timeframe_minutos "
              "(ej. rango de 15 con velas de 5).")
    horas["fin_rango"] = horas["inicio"] + rg["minutos"]
    if not horas["fin_rango"] < horas["hora_limite"] <= horas["hora_cierre"]:
        error("debe cumplirse: fin del rango < entrada.hora_limite <= salida.hora_cierre.")
    st = r["stop"]
    usados = [st["tipo"]] + ([st["respaldo"]] if st["tipo"] == "ema" else [])
    for tipo in usados:
        if tipo in ("puntos", "multiplo_rango") and not st["valor"] > 0:
            error(f"el stop '{tipo}' necesita stop.valor mayor que 0.")
    if st["min_puntos"] and st["max_puntos"] and st["min_puntos"] > st["max_puntos"]:
        error("stop.min_puntos no puede ser mayor que stop.max_puntos.")
    if r["salida"]["objetivo_tipo"] != "ninguno" and not r["salida"]["objetivo_valor"] > 0:
        error("salida.objetivo_valor debe ser mayor que 0 (o usa objetivo_tipo = 'ninguno').")
    if r["trailing"]["tipo"] in ("puntos", "R") and not r["trailing"]["distancia"] > 0:
        error("trailing.distancia debe ser mayor que 0 para ese tipo de trailing.")
    if not r["parcial"]["porcentaje"] < 100:
        error("parcial.porcentaje debe ser menor que 100 (0 = sin salida parcial).")
    malos = set(r["filtros"]["dias_semana"]) - set(DIAS)
    if malos or not r["filtros"]["dias_semana"]:
        error(f"filtros.dias_semana tiene días no válidos ({sorted(malos)}). "
              f"Usa: {', '.join(DIAS[:5])}.")
    try:
        horas["excluir"] = {pd.Timestamp(f) for f in r["filtros"]["excluir_fechas"]}
    except Exception:
        error("filtros.excluir_fechas debe ser una lista de fechas \"AAAA-MM-DD\".")
    return r, horas


def huella(reglas):
    """Identificador de la variante: cambia si cambia cualquier parámetro (no los comentarios).
    Los parámetros de EMA se ignoran si la variante no usa EMA, para que agregar esa opción
    no cambie la huella de las variantes que ya existían.
    (2026-10-23: con trailing EMA ya no se entra si la EMA está del lado equivocado; ninguna
    corrida anterior usaba trailing EMA, así que la huella no cambió. Ídem: la salida parcial
    ahora cierra solo contratos enteros; ninguna variante activa usaba parcial.)
    Los parámetros agregados después (NUEVOS) no cuentan mientras tengan su valor por defecto,
    así las variantes anteriores conservan su huella."""
    sin_textos = {k: v for k, v in reglas.items() if k not in ("nombre", "descripcion")}
    for (seccion, clave), defecto in NUEVOS.items():
        if seccion in sin_textos and sin_textos[seccion].get(clave) == defecto:
            sin_textos[seccion] = {k: v for k, v in sin_textos[seccion].items() if k != clave}
    if sin_textos.get("stop", {}).get("tipo") != "ema" and "stop" in sin_textos:
        sin_textos["stop"] = {k: v for k, v in sin_textos["stop"].items() if k != "respaldo"}
    if "ema" not in (sin_textos.get("stop", {}).get("tipo"),
                     sin_textos.get("trailing", {}).get("tipo")) \
            and sin_textos.get("entrada", {}).get("tipo") != "cierre_ema":
        sin_textos.pop("ema", None)
    return hashlib.sha256(json.dumps(sin_textos, sort_keys=True).encode()).hexdigest()[:12]


# parámetros agregados después (2026-10-23 y 2026-10-24) y su valor por defecto (ver huella)
NUEVOS = {("stop", "min_puntos"): 0, ("stop", "si_menor_min"): "saltar",
          ("retesteo", "ruptura_previa_en"): "minuto", ("trailing", "timeframe_minutos"): 0}


def huella_reglas(reglas):
    """Como huella, pero sin costos ni tamaño de posición: identifica las reglas de trading.
    El candado del out-of-sample usa esta, para que cambiar la cuenta o el riesgo no permita
    volver a mirar el out-of-sample de las mismas reglas."""
    return huella({k: v for k, v in reglas.items() if k not in ("costos", "riesgo")})


# ---------------------------------------------------------------- datos

def cargar_dias(hasta):
    """Velas de 1 minuto hasta la fecha 'hasta' (no lee meses posteriores) y datos por día."""
    archivos = [a for a in sorted(PROCESSED.glob("barras_1m_*.csv")) if a.stem[-7:] <= hasta[:7]]
    if not archivos:
        raise SystemExit("No hay velas de 1 minuto: ejecuta scripts\\datos\\construir_barras.py")
    v = pd.concat([leer_barras(a) for a in archivos], ignore_index=True).sort_values("ts_utc")
    hora = v["ts_ny"].dt.tz_localize(None)
    v["fecha"] = hora.dt.normalize()
    v["minuto"] = hora.dt.hour * 60 + hora.dt.minute
    v = v[v["fecha"] <= pd.Timestamp(hasta)]

    arreglos = {
        fecha: (d["minuto"].to_numpy(), d["open"].to_numpy(), d["high"].to_numpy(),
                d["low"].to_numpy(), d["close"].to_numpy(), d["volumen"].to_numpy(float),
                d["symbol"].to_numpy())
        for fecha, d in v.groupby("fecha", sort=True)
    }

    rth = v[(v["minuto"] >= INICIO_RTH) & (v["minuto"] < FIN_RTH)].groupby("fecha")
    info = pd.DataFrame({"alto": rth["high"].max(), "bajo": rth["low"].min(),
                         "cierre": rth["close"].last(), "ultimo": rth["minuto"].max()})
    info["ancho_dia"] = info["alto"] - info["bajo"]
    info["reducido"] = info["ultimo"] < TARDE
    feriados = CalendarioNYSE().holidays(info.index.min(), info.index.max())
    info["feriado"] = info.index.isin(feriados)
    habiles = info[~info["feriado"]]
    info["cierre_previo"] = habiles["cierre"].shift().reindex(info.index)
    VELAS.clear()
    VELAS.update(df=v[["ts_utc", "fecha", "open", "close", "symbol"]].reset_index(drop=True),
                 emas={})
    return arreglos, info


VELAS = {}  # velas de 1 minuto continuas (24 h) de la última carga, para calcular EMAs
CALENTAMIENTO = 3  # la EMA es válida después de CALENTAMIENTO x periodos velas


def ema_por_dia(periodos, tf):
    """EMA de 'periodos' velas de 'tf' minutos (24 horas), ajustada por los rolls de contrato.
    Devuelve {fecha: arreglo alineado con las velas de 1 minuto del día}: en cada minuto, la
    EMA de la última vela de tf minutos YA CERRADA (nunca mira el futuro). NaN mientras se
    calienta (menos de CALENTAMIENTO x periodos velas desde el inicio de los datos)."""
    if (periodos, tf) in VELAS["emas"]:
        return VELAS["emas"][(periodos, tf)]
    v = VELAS["df"]
    # Ajuste por roll: los precios de contratos anteriores se llevan al nivel del último
    # contrato sumando los saltos de los rolls posteriores (como un continuo ajustado).
    roll = (v["symbol"] != v["symbol"].shift()).to_numpy(copy=True)
    roll[0] = False
    salto = np.where(roll, v["open"].to_numpy() - v["close"].shift().to_numpy(), 0.0)
    ajuste = np.cumsum(salto[::-1])[::-1] - salto  # suma de los saltos POSTERIORES a cada vela
    cierre_aj = v["close"].to_numpy() + ajuste

    ts = v["ts_utc"]
    if ts.dt.tz is not None:
        ts = ts.dt.tz_convert(None)  # UTC sin zona: comparaciones rápidas y sin cambios de horario
    vela = ts.dt.floor(f"{tf}min")
    cierres = pd.Series(cierre_aj).groupby(vela.to_numpy()).last()
    ema = cierres.ewm(span=periodos, adjust=False).mean().to_numpy(copy=True)
    ema[: CALENTAMIENTO * periodos - 1] = np.nan
    fin_vela = (cierres.index + pd.Timedelta(minutes=tf)).to_numpy()
    # para cada minuto: la última vela de tf minutos que ya terminó antes de que empiece
    k = np.searchsorted(fin_vela, ts.to_numpy(), side="right") - 1
    valor = np.where(k >= 0, ema[np.maximum(k, 0)], np.nan) - ajuste  # vuelve al contrato del minuto

    resultado = {fecha: valor[idx] for fecha, idx in v.groupby("fecha", sort=True).indices.items()}
    VELAS["emas"][(periodos, tf)] = resultado
    return resultado


# ---------------------------------------------------------------- simulación

def redondear(precio):
    return round(precio / TICK) * TICK


def hhmm(minuto):
    return f"{minuto // 60:02d}:{minuto % 60:02d}"


def simular_dia(r, horas, fecha, dia, info_dia, ema=None):
    """Simula un día. Devuelve la lista de trades (diccionarios). 'ema' (opcional) es la EMA
    disponible en cada minuto del día (ver ema_por_dia)."""
    m, o, h, l, c, vol, simbolo = dia
    rg, e, rt, st, be, tr, sal, par, fl, ej, co, rs = (
        r[k] for k in ("rango", "entrada", "retesteo", "stop", "breakeven", "trailing",
                       "salida", "parcial", "filtros", "ejecucion", "costos", "riesgo"))
    ini, fin_r = horas["inicio"], horas["fin_rango"]
    limite, cierre, tf = horas["hora_limite"], horas["hora_cierre"], e["timeframe_minutos"]
    tf_tr = tr["timeframe_minutos"] or tf  # velas del trailing "velas" (0 = las de la entrada)
    # la ruptura solo cuenta en las primeras "velas_para_ruptura" velas de timeframe tras el rango
    fin_ruptura = fin_r + e["velas_para_ruptura"] * tf if e["velas_para_ruptura"] else np.inf

    # --- rango de apertura y filtros que dependen de él
    en_rango = (m >= ini) & (m < fin_r)
    if not en_rango.any() or m[en_rango][0] != ini:
        return []
    r_hi, r_lo = h[en_rango].max(), l[en_rango].min()
    ancho = r_hi - r_lo
    if ancho <= 0:
        return []
    if (rg["ancho_min_puntos"] and ancho < rg["ancho_min_puntos"]) or \
            (rg["ancho_max_puntos"] and ancho > rg["ancho_max_puntos"]):
        return []
    if rg["ancho_min_vs_promedio"] or rg["ancho_max_vs_promedio"]:
        promedio = info_dia["ancho_promedio"]
        if not promedio > 0:  # también descarta "sin dato" (primeros días)
            return []
        relativo = ancho / promedio
        if (rg["ancho_min_vs_promedio"] and relativo < rg["ancho_min_vs_promedio"]) or \
                (rg["ancho_max_vs_promedio"] and relativo > rg["ancho_max_vs_promedio"]):
            return []
    gap = o[np.argmax(en_rango)] - info_dia["cierre_previo"]
    if fl["gap_min_puntos"] or fl["gap_max_puntos"] or fl["a_favor_de"] == "gap":
        if np.isnan(gap):
            return []
        if (fl["gap_min_puntos"] and abs(gap) < fl["gap_min_puntos"]) or \
                (fl["gap_max_puntos"] and abs(gap) > fl["gap_max_puntos"]):
            return []

    # VWAP aproximado desde el inicio del rango (precio típico de cada vela de 1 minuto)
    desde_ini = m >= ini
    tipico = np.where(desde_ini, (h + l + c) / 3 * vol, 0).cumsum()
    volumen = np.where(desde_ini, vol, 0).cumsum()
    vwap = np.divide(tipico, volumen, out=np.full(len(m), np.nan), where=volumen > 0)

    slip_e = ej["slippage_entrada_ticks"] * TICK
    slip_s = ej["slippage_salida_ticks"] * TICK
    traspaso = ej["limite_requiere_traspaso_ticks"] * TICK
    buffer = e["buffer_ticks"] * TICK
    peor = ej["misma_vela"] == "peor"

    trades = []
    estado = dict(pos=None, pendiente=None, entrada_mercado=None, salida_mercado=False,
                  entradas=0, ultima_dir=0)
    bloques = []  # velas de timeframe ya cerradas después del rango: [open, high, low, close]
    # retesteo_cierre: ¿el precio ya salió del rango? y ¿la vela de timeframe en curso tocó el borde?
    salio, toco = {1: False, -1: False}, {1: False, -1: False}
    salio_antes = dict(salio)  # salió en una vela de timeframe ANTERIOR (ruptura_previa_en = "vela")
    toque_retesteo = rt["offset_ticks"] * TICK

    def permitida(d, i_previa):
        if e["direccion"] == "solo_largos" and d < 0 or e["direccion"] == "solo_cortos" and d > 0:
            return False
        if not e["permitir_misma_direccion"] and d == estado["ultima_dir"]:
            return False
        if fl["a_favor_de"] == "vwap" and not d * (c[i_previa] - vwap[i_previa]) > 0:
            return False
        if fl["a_favor_de"] == "gap" and not d * gap > 0:
            return False
        return True

    def abrir(d, precio, i):
        """Abre una posición. Devuelve False si el trade se descarta (riesgo inválido)."""
        if estado["entradas"] >= e["max_trades_dia"]:
            return False
        # trailing EMA con la EMA del lado equivocado (ej. largo bajo la EMA): el trailing
        # pondría el stop al otro lado del precio y cerraría el trade al instante -> no entra
        if tr["tipo"] == "ema" and ema is not None and not np.isnan(ema[i]) \
                and not d * (precio - ema[i]) > 0:
            return False
        sb = st["buffer_ticks"] * TICK

        def nivel_stop(tipo):
            if tipo == "rango_opuesto":
                return (r_lo if d > 0 else r_hi) - d * sb
            if tipo == "medio_rango":
                return (r_hi + r_lo) / 2 - d * sb
            if tipo == "puntos":
                return precio - d * st["valor"]
            if tipo == "multiplo_rango":
                return precio - d * st["valor"] * ancho
            if tipo == "ema":  # EMA de la última vela cerrada; NaN si aún no está lista
                return ema[i] - d * sb if ema is not None else np.nan
            # vela_senal: extremo de la última vela de timeframe cerrada
            _, bh, bl, _ = bloques[-1] if bloques else (0, r_hi, r_lo, 0)
            return (bl if d > 0 else bh) - d * sb

        usado = st["tipo"]
        stop = nivel_stop(usado)
        if usado == "ema" and not d * (precio - stop) > 0:  # no lista o del lado equivocado
            usado = st["respaldo"]
            if usado == "no_operar":
                return False
            stop = nivel_stop(usado)
        stop = redondear(stop)
        riesgo = d * (precio - stop)
        if riesgo <= 0:
            return False
        if st["min_puntos"] and riesgo < st["min_puntos"]:  # stop demasiado cerca de la entrada
            if st["si_menor_min"] == "saltar":
                return False
            stop = redondear(precio - d * st["min_puntos"])
            riesgo = d * (precio - stop)
        if st["max_puntos"] and riesgo > st["max_puntos"]:
            if st["si_excede_max"] == "saltar":
                return False
            stop = redondear(precio - d * st["max_puntos"])
            riesgo = d * (precio - stop)
        if rs["modo"] == "riesgo_usd":  # cuántos contratos caben en el riesgo en USD
            contratos = max(int(rs["riesgo_usd"] // (riesgo * co["valor_punto_usd"])),
                            rs["minimo_contratos"])
            if contratos < 1:
                return False  # ni 1 contrato cabe en el riesgo y minimo_contratos = 0
        else:
            contratos = co["contratos"]
        tipo_obj, valor_obj = sal["objetivo_tipo"], sal["objetivo_valor"]
        distancia = {"R": valor_obj * riesgo, "puntos": valor_obj,
                     "multiplo_rango": valor_obj * ancho}.get(tipo_obj)
        estado["entradas"] += 1
        estado["ultima_dir"] = d
        estado["pos"] = dict(
            d=d, entrada=precio, stop=stop, stop_ini=stop, riesgo=riesgo, i_entrada=i,
            contratos=contratos, stop_usado=usado,
            objetivo=None if distancia is None else redondear(precio + d * distancia),
            parcial=redondear(precio + d * par["en_R"] * riesgo) if par["porcentaje"] else None,
            restante=1.0, realizado=0.0, mfe=0.0, mae=0.0, motivo_stop="stop",
            be_pendiente=False,
        )
        return True

    def cerrar(precio, i, motivo):
        p = estado["pos"]
        d, riesgo = p["d"], p["riesgo"]
        movimiento = d * (precio - p["entrada"])  # la vela de salida también cuenta para MFE/MAE
        p["mfe"], p["mae"] = max(p["mfe"], movimiento), min(p["mae"], movimiento)
        puntos = p["realizado"] + p["restante"] * movimiento
        n = p["contratos"]
        bruto = puntos * co["valor_punto_usd"] * n
        trades.append(dict(
            fecha=fecha.date().isoformat(), contrato=simbolo[p["i_entrada"]],
            direccion="largo" if d > 0 else "corto", hora_entrada=hhmm(m[p["i_entrada"]]),
            precio_entrada=p["entrada"], stop_inicial=p["stop_ini"], stop_usado=p["stop_usado"],
            objetivo="" if p["objetivo"] is None else p["objetivo"], hora_salida=hhmm(m[i]),
            precio_salida=round(p["entrada"] + d * puntos, 4), motivo_salida=motivo,
            puntos=round(puntos, 4), R=round(puntos / riesgo, 3), contratos=n,
            riesgo_usd=round(riesgo * co["valor_punto_usd"] * n, 2), usd_bruto=round(bruto, 2),
            usd_neto=round(bruto - co["comision_ida_vuelta_usd"] * n, 2),
            rango_alto=r_hi, rango_bajo=r_lo, ancho_rango=ancho,
            mfe_R=round(p["mfe"] / riesgo, 3), mae_R=round(p["mae"] / riesgo, 3),
            duracion_min=int(m[i] - m[p["i_entrada"]]),
        ))
        estado["pos"] = None

    def mover_stop(x, motivo):
        """x en coordenadas 'a favor' (d * precio). El stop solo se acerca, nunca se aleja."""
        p = estado["pos"]
        if x > p["d"] * p["stop"]:
            p["stop"] = redondear(p["d"] * x)
            p["motivo_stop"] = motivo

    def gestionar(i, recien):
        """Revisa stop, salida parcial y objetivo dentro de la vela i."""
        p = estado["pos"]
        d = p["d"]
        f_hi, f_lo = max(d * h[i], d * l[i]), min(d * h[i], d * l[i])
        f_o = d * (p["entrada"] if recien else o[i])  # si entró en esta vela, parte del precio de entrada
        for evento in (("stop", "parcial", "objetivo") if peor else ("parcial", "objetivo", "stop")):
            if evento == "stop" and f_lo <= d * p["stop"]:
                cerrar(d * (min(f_o, d * p["stop"]) - slip_s), i, p["motivo_stop"])
                return
            if evento == "parcial" and p["parcial"] is not None \
                    and f_hi >= d * p["parcial"] + traspaso:
                precio = d * max(f_o, d * p["parcial"])
                # solo contratos enteros (redondeado hacia abajo): con 1 contrato no hay parcial,
                # pero el breakeven tras la parcial se aplica igual
                n = p["contratos"]
                fraccion = math.floor(n * par["porcentaje"] / 100) / n
                p["realizado"] += fraccion * d * (precio - p["entrada"])
                p["restante"] -= fraccion
                p["parcial"] = None
                p["be_pendiente"] = par["breakeven_tras_parcial"]  # rige desde la vela siguiente
            if evento == "objetivo" and p["objetivo"] is not None \
                    and f_hi >= d * p["objetivo"] + traspaso:
                cerrar(d * max(f_o, d * p["objetivo"]), i, "objetivo")
                return

    def fin_de_vela(i):
        """Al cierre de la vela de 1 minuto: MFE/MAE, breakeven y trailing (rigen desde la siguiente)."""
        p = estado["pos"]
        d, riesgo, entrada = p["d"], p["riesgo"], p["d"] * p["entrada"]
        p["mfe"] = max(p["mfe"], max(d * h[i], d * l[i]) - entrada)
        p["mae"] = min(p["mae"], min(d * h[i], d * l[i]) - entrada)
        if p["be_pendiente"] or (be["activar_en_R"] and p["mfe"] >= be["activar_en_R"] * riesgo):
            mover_stop(entrada + be["offset_ticks"] * TICK, "breakeven")
            p["be_pendiente"] = False
        if tr["tipo"] in ("puntos", "R") and p["mfe"] >= tr["activar_en_R"] * riesgo:
            distancia = tr["distancia"] * (riesgo if tr["tipo"] == "R" else 1)
            mover_stop(entrada + p["mfe"] - distancia, "trailing")

    def senal(d):
        """Hay ruptura en dirección d: entra (breakout) o espera el retesteo."""
        if e["tipo"] == "retesteo":
            nivel = (r_hi if d > 0 else r_lo) + d * rt["offset_ticks"] * TICK
            estado["pendiente"] = dict(d=d, nivel=nivel, velas=0)
            return False
        return True

    def trailing_velas(lista):
        """Trailing "velas": stop bajo el mínimo (largos) / sobre el máximo (cortos) de las
        últimas tr["velas"] velas cerradas de la lista."""
        p = estado["pos"]
        if p and tr["tipo"] == "velas" and p["mfe"] >= tr["activar_en_R"] * p["riesgo"] \
                and len(lista) >= tr["velas"]:
            d = p["d"]
            piso = min(min(d * b[1], d * b[2]) for b in lista[-tr["velas"]:])
            mover_stop(piso - st["buffer_ticks"] * TICK, "trailing")

    def cierre_de_bloque(bloque, i_ultima):
        """Al cerrar una vela de timeframe (se ejecuta en la apertura de la siguiente)."""
        bloques.append(bloque)
        cierre_vela = bloque[3]
        dentro = r_lo <= cierre_vela <= r_hi
        p = estado["pos"]
        if p:
            if sal["salir_si_cierra_dentro"] and dentro:
                estado["salida_mercado"] = True
            if tf_tr == tf:
                trailing_velas(bloques)
        pend = estado["pendiente"]
        if pend:
            pend["velas"] += 1
            if (rt["cancelar_si_cierra_dentro"] and dentro) or pend["velas"] > rt["max_velas_espera"]:
                estado["pendiente"] = None
        if ((e["confirmacion"] == "cierre" or e["tipo"] == "retesteo_cierre")
                and e["tipo"] != "cierre_ema"
                and not p and not estado["pendiente"] and estado["entrada_mercado"] is None
                and estado["entradas"] < e["max_trades_dia"] and m[i_ultima] < fin_ruptura):
            d = 1 if cierre_vela > r_hi + buffer else -1 if cierre_vela < r_lo - buffer else 0
            if e["tipo"] == "retesteo_cierre" and d and not toco[d]:
                d = 0  # cerró fuera, pero la vela no tocó el borde del rango
            if d and permitida(d, i_ultima) and senal(d):
                estado["entrada_mercado"] = d

    if e["tipo"] == "cierre_ema":
        # señal única: la vela del rango (ej. 09:30-09:35) cierra sobre la EMA -> largo, bajo la
        # EMA -> corto. La EMA del primer minuto tras el rango ya incluye esa vela (ema.timeframe
        # = rango.minutos). Se entra a mercado en la apertura de ese minuto. Igual a la EMA o
        # EMA sin datos: no se opera.
        despues = np.nonzero(m >= fin_r)[0]
        if len(despues) and m[despues[0]] == fin_r and ema is not None \
                and not np.isnan(ema[despues[0]]):
            i_sig = despues[0]
            cierre_rango = c[en_rango][-1]
            d = 1 if cierre_rango > ema[i_sig] else -1 if cierre_rango < ema[i_sig] else 0
            if d and permitida(d, i_sig - 1):
                estado["entrada_mercado"] = d

    # trailing "velas" con velas de otro tamaño que las de la entrada (trailing.timeframe_minutos):
    # velas alineadas al inicio del rango (ej. 15 min: 09:30-09:45, 09:45-10:00...)
    bloques_tr, id_tr = [], None
    bloque, id_bloque, ultima = None, None, None
    for i in np.nonzero(m >= fin_r)[0]:
        t = m[i]
        id_actual = (t - ini) // tf
        if bloque is not None and id_actual != id_bloque:
            cierre_de_bloque(bloque, i - 1)
            bloque = None
        if tf_tr != tf:
            id_t = (t - ini) // tf_tr
            if id_tr is not None and id_t != id_tr:  # cerró una vela de tf_tr minutos
                desde = ini + id_tr * tf_tr
                sel = (m >= desde) & (m < desde + tf_tr)
                bloques_tr.append([None, h[sel].max(), l[sel].min(), None])
                trailing_velas(bloques_tr)
            id_tr = id_t
        if t >= cierre:
            if estado["pos"]:
                cerrar(o[i] - estado["pos"]["d"] * slip_s, i, "hora_cierre")
            break
        ultima = i
        recien = False
        if estado["pos"] and estado["salida_mercado"]:
            cerrar(o[i] - estado["pos"]["d"] * slip_s, i, "cierre_dentro_rango")
        estado["salida_mercado"] = False

        d = estado["entrada_mercado"]
        estado["entrada_mercado"] = None
        if d and not estado["pos"] and t < limite:
            recien = abrir(d, o[i] + d * slip_e, i)

        pend = estado["pendiente"]
        if pend and not estado["pos"]:
            if t >= limite:
                estado["pendiente"] = None
            elif min(pend["d"] * h[i], pend["d"] * l[i]) <= pend["d"] * pend["nivel"] - traspaso:
                d = pend["d"]
                estado["pendiente"] = None
                recien = abrir(d, d * min(d * o[i], d * pend["nivel"]), i)

        if (e["confirmacion"] == "toque" and e["tipo"] not in ("retesteo_cierre", "cierre_ema")
                and not estado["pos"] and not estado["pendiente"]
                and not recien and estado["entradas"] < e["max_trades_dia"] and t < limite
                and t < fin_ruptura):
            sube, baja = h[i] >= r_hi + buffer, l[i] <= r_lo - buffer
            if sube and baja:  # tocó ambos lados en la misma vela: el más cercano a la apertura
                sube = (r_hi + buffer - o[i]) <= (o[i] - (r_lo - buffer))
                baja = not sube
            d = 1 if sube else -1 if baja else 0
            if d and permitida(d, i - 1) and senal(d):
                gatillo = (r_hi if d > 0 else r_lo) + d * buffer
                recien = abrir(d, d * max(d * o[i], d * gatillo) + d * slip_e, i)

        p = estado["pos"]
        if p and tr["tipo"] == "ema" and not recien and ema is not None \
                and not np.isnan(ema[i]) and p["mfe"] >= tr["activar_en_R"] * p["riesgo"]:
            # el stop sigue a la EMA de la última vela cerrada (solo a favor)
            mover_stop(p["d"] * ema[i] - st["buffer_ticks"] * TICK, "trailing")
        if estado["pos"]:
            gestionar(i, recien)
        if estado["pos"]:
            fin_de_vela(i)

        if e["tipo"] == "retesteo_cierre":
            if bloque is None:  # empieza una vela de timeframe nueva
                toco[1] = toco[-1] = False
                salio_antes.update(salio)
            previa = salio_antes if rt["ruptura_previa_en"] == "vela" else salio
            for d, borde in ((1, r_hi), (-1, r_lo)):
                # el toque cuenta solo si el precio salió del rango en un minuto ANTERIOR
                # (o en una vela de timeframe anterior, con ruptura_previa_en = "vela")
                if min(d * h[i], d * l[i]) <= d * borde + toque_retesteo and \
                        (previa[d] or not rt["requiere_ruptura_previa"]):
                    toco[d] = True
                if max(d * h[i], d * l[i]) >= d * borde + buffer:
                    salio[d] = True

        if bloque is None:
            bloque, id_bloque = [o[i], h[i], l[i], c[i]], id_actual
        else:
            bloque[1], bloque[2], bloque[3] = max(bloque[1], h[i]), min(bloque[2], l[i]), c[i]
    else:
        if estado["pos"] and ultima is not None:  # el día terminó antes de la hora de cierre
            cerrar(c[ultima], ultima, "fin_de_datos")
    return trades


def correr(r, horas, arreglos, info, periodo):
    """Corre una variante en un periodo. Devuelve (trades DataFrame, días evaluados)."""
    desde, hasta = map(pd.Timestamp, PERIODOS[periodo])
    fl = r["filtros"]
    completos = info[~info["feriado"] & ~info["reducido"]]
    promedio = (completos["ancho_dia"].rolling(r["rango"]["dias_promedio"]).mean().shift()
                .reindex(info.index).ffill())
    usa_ema = (r["stop"]["tipo"] == "ema" or r["trailing"]["tipo"] == "ema"
               or r["entrada"]["tipo"] == "cierre_ema")
    emas = ema_por_dia(r["ema"]["periodos"], r["ema"]["timeframe_minutos"]) if usa_ema else {}
    trades, evaluados = [], []
    for fecha, dia in arreglos.items():
        if not desde <= fecha <= hasta or fecha not in info.index:
            continue
        datos = info.loc[fecha]
        if (DIAS[fecha.weekday()] not in fl["dias_semana"]
                or (fl["excluir_feriados"] and datos["feriado"])
                or (fl["excluir_dias_reducidos"] and datos["reducido"])
                or fecha in horas["excluir"]):
            continue
        evaluados.append(fecha)
        datos = datos.copy()
        datos["ancho_promedio"] = promedio[fecha]
        trades += simular_dia(r, horas, fecha, dia, datos, emas.get(fecha))
    return pd.DataFrame(trades, columns=COLUMNAS_TRADES), evaluados


# ---------------------------------------------------------------- resultados

def resumir(t, evaluados, capital):
    res = dict.fromkeys(COLUMNAS_RESUMEN, 0)
    res.update(dias_evaluados=len(evaluados), capital_inicial_usd=capital,
               saldo_final_usd=capital, saldo_minimo_usd=capital)
    if t.empty:
        return res
    neto = t["usd_neto"]
    ganan, pierden = neto[neto > 0], neto[neto <= 0]
    curva = neto.cumsum().to_numpy()
    pico = np.maximum.accumulate(np.maximum(curva, 0))
    racha = max_racha = 0
    for x in neto:
        racha = racha + 1 if x <= 0 else 0
        max_racha = max(max_racha, racha)
    diario = neto.groupby(t["fecha"]).sum().reindex(
        [f.date().isoformat() for f in evaluados], fill_value=0)
    desvio = diario.std()
    largos, cortos = t["direccion"] == "largo", t["direccion"] == "corto"
    res.update(
        trades=len(t), dias_con_trade=t["fecha"].nunique(),
        pct_acierto=round(100 * len(ganan) / len(t), 1), neto_usd=round(neto.sum(), 2),
        promedio_usd=round(neto.mean(), 2),
        ganancia_prom_usd=round(ganan.mean(), 2) if len(ganan) else 0,
        perdida_prom_usd=round(pierden.mean(), 2) if len(pierden) else 0,
        profit_factor=round(ganan.sum() / -pierden.sum(), 2) if pierden.sum() < 0 else math.inf,
        expectativa_R=round(t["R"].mean(), 3), max_drawdown_usd=round((curva - pico).min(), 2),
        racha_perdedora=max_racha,
        sharpe_anual=round(diario.mean() / desvio * math.sqrt(252), 2) if desvio > 0 else 0,
        largos=int(largos.sum()), neto_largos_usd=round(neto[largos].sum(), 2),
        cortos=int(cortos.sum()), neto_cortos_usd=round(neto[cortos].sum(), 2),
    )
    saldo = capital + curva  # saldo de la cuenta después de cada trade
    pico_saldo = np.maximum.accumulate(np.maximum(saldo, capital))
    res.update(
        saldo_final_usd=round(saldo[-1], 2), retorno_pct=round(100 * curva[-1] / capital, 1),
        max_drawdown_pct=round(100 * ((saldo - pico_saldo) / pico_saldo).min(), 1),
        saldo_minimo_usd=round(min(saldo.min(), capital), 2),
        riesgo_prom_usd=round(t["riesgo_usd"].mean(), 2),
        riesgo_max_usd=round(t["riesgo_usd"].max(), 2),
    )
    return res


def leer_registro():
    if not REGISTRO.exists():
        return pd.DataFrame(columns=["fecha_hora", "periodo", "variante", "archivo", "huella",
                                     "huella_reglas"])
    return pd.read_csv(REGISTRO, dtype=str)


def anotar(periodo, variante, archivo, marca, marca_reglas, res):
    columnas = ["fecha_hora", "periodo", "variante", "archivo", "huella", "huella_reglas",
                *COLUMNAS_RESUMEN]
    if REGISTRO.exists():  # si se agregaron métricas nuevas, completa el registro antiguo
        anterior = pd.read_csv(REGISTRO, dtype=str)
        if list(anterior.columns) != columnas:
            anterior.reindex(columns=columnas).to_csv(REGISTRO, index=False)
    nuevo = not REGISTRO.exists()
    with REGISTRO.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(columnas)
        w.writerow([datetime.now().isoformat(timespec="seconds"), periodo, variante,
                    archivo.name, marca, marca_reglas, *(res[k] for k in COLUMNAS_RESUMEN)])


def guardar_reglas(archivo, r):
    """Guarda una copia de las reglas de la variante (una vez por huella) en
    resultados/reglas_probadas: el archivo tal cual (.toml) y todos los parámetros ya
    completados con los valores por defecto (.json). Así siempre se sabe qué se probó,
    aunque después se cambie o borre el archivo, o cambien los valores por defecto."""
    REGLAS_PROBADAS.mkdir(parents=True, exist_ok=True)
    base = REGLAS_PROBADAS / f"{r['nombre']}_{huella(r)}"
    if base.with_suffix(".json").exists():
        return
    base.with_suffix(".toml").write_text(archivo.read_text(encoding="utf-8"), encoding="utf-8")
    datos = {"variante": r["nombre"], "archivo": archivo.name, "huella": huella(r),
             "huella_reglas": huella_reglas(r),
             "guardado": datetime.now().isoformat(timespec="seconds"), "reglas": r}
    base.with_suffix(".json").write_text(json.dumps(datos, indent=2, ensure_ascii=False),
                                         encoding="utf-8")


def reglas_probadas(variante, marca):
    """Parámetros completos guardados de una variante y huella (None si no hay copia)."""
    ruta = REGLAS_PROBADAS / f"{variante}_{marca}.json"
    return json.loads(ruta.read_text(encoding="utf-8"))["reglas"] if ruta.exists() else None


def reglas_ya_en_oos(registro):
    """Huellas de reglas (sin costos ni riesgo) que ya vieron los meses out-of-sample
    (corridas out-of-sample o completo)."""
    oos = registro[registro["periodo"].isin(["out-of-sample", "completo"])]
    return set(oos.get("huella_reglas", pd.Series(dtype=str)).dropna())


def estrategia_de(archivo):
    """Subcarpeta de reglas a la que pertenece un archivo (ej. "ORB"); "otras" si el
    archivo está fuera de esas subcarpetas. Decide la carpeta de resultados."""
    padre = Path(archivo).resolve().parent
    return padre.name if padre.parent == REGLAS.resolve() else "otras"


def carpeta_resultados(estrategia, periodo):
    return RESULTADOS / estrategia / periodo


def main():
    parser = argparse.ArgumentParser(description="Backtest de variantes sobre NQ.")
    parser.add_argument("--reglas", nargs="+",
                        help="Archivos .toml (por defecto: todas las subcarpetas de reglas)")
    parser.add_argument("--estrategia",
                        help="Subcarpeta de reglas a correr completa (ej. ORB)")
    parser.add_argument("--out-of-sample", action="store_true",
                        help="Usa el periodo bloqueado 2026-04-01 a 2026-09-30 (una vez por variante)")
    parser.add_argument("--repetir-oos", action="store_true",
                        help="Permite repetir out-of-sample con una variante ya probada ahí")
    parser.add_argument("--completo", action="store_true",
                        help="Usa los 12 meses (incluye out-of-sample); solo variantes ya probadas ahí")
    parser.add_argument("--acepto-gastar-oos", action="store_true",
                        help="Con --completo: permite variantes que no pasaron por out-of-sample. "
                             "Quedan marcadas y ya no podrán hacer un out-of-sample limpio")
    args = parser.parse_args()
    if args.acepto_gastar_oos and not args.completo:
        sys.exit("--acepto-gastar-oos solo tiene sentido junto con --completo.")
    if args.repetir_oos and not args.out_of_sample:
        sys.exit("--repetir-oos solo tiene sentido junto con --out-of-sample.")
    if args.completo and args.out_of_sample:
        sys.exit("Usa --out-of-sample o --completo, no los dos.")
    periodo = ("out-of-sample" if args.out_of_sample else "completo" if args.completo
               else "in-sample")

    if args.reglas and args.estrategia:
        sys.exit("Usa --reglas o --estrategia, no los dos.")
    if args.estrategia and not (REGLAS / args.estrategia).is_dir():
        sys.exit(f"No existe la carpeta estrategias\\{args.estrategia}.")
    if args.estrategia in OTROS_MOTORES:
        sys.exit(f"{args.estrategia} tiene su propio motor: {OTROS_MOTORES[args.estrategia]}")
    patron = f"{args.estrategia}/*.toml" if args.estrategia else "*/*.toml"
    archivos = ([Path(a) for a in args.reglas] if args.reglas
                else [a for a in sorted(REGLAS.glob(patron)) if not a.name.startswith("_")
                      and a.parent.name not in OTROS_MOTORES])
    if not archivos:
        sys.exit("No hay archivos de reglas (.toml) en las subcarpetas de reglas.")
    for a in archivos:
        if not a.exists():
            sys.exit(f"No existe el archivo {a}.")
    variantes = [(a, *cargar_reglas(a)) for a in archivos]
    nombres = [r["nombre"] for _, r, _ in variantes]
    repetidos = {n for n in nombres if nombres.count(n) > 1}
    if repetidos:
        sys.exit(f"Hay variantes con el mismo nombre: {', '.join(sorted(repetidos))}.")

    registro = leer_registro()
    if args.out_of_sample:
        print("=" * 70)
        print(" OUT-OF-SAMPLE (2026-04-01 a 2026-09-30): la prueba final. Úsala UNA vez,")
        print(" con las variantes ya elegidas y el criterio de aprobación ya decidido.")
        print("=" * 70)
        ya = reglas_ya_en_oos(registro)
        for a, r, _ in variantes:
            if huella_reglas(r) in ya and not args.repetir_oos:
                sys.exit(f"Las reglas de '{r['nombre']}' ({a.name}) ya se probaron en "
                         "out-of-sample (aunque cambien costos o riesgo, son las mismas reglas). "
                         "Repetir le quita valor a la prueba; si de verdad hace falta, "
                         "agrega --repetir-oos.")
    if args.completo:
        ya = reglas_ya_en_oos(registro)
        nuevas = [r["nombre"] for _, r, _ in variantes if huella_reglas(r) not in ya]
        if nuevas and not args.acepto_gastar_oos:
            sys.exit(f"Estas variantes no se han probado en out-of-sample: {', '.join(nuevas)}. "
                     "El periodo completo incluye esos meses: córrelas primero con "
                     "--out-of-sample, o agrega --acepto-gastar-oos si decides usar los 12 meses "
                     "sabiendo que después no tendrán un out-of-sample limpio.")
        if nuevas:
            print(f"AVISO: {', '.join(nuevas)} ven ahora los meses out-of-sample por primera vez; "
                  "quedan registradas y ya no podrán correr --out-of-sample sin --repetir-oos.")

    print(f"Periodo {periodo}: {PERIODOS[periodo][0]} a {PERIODOS[periodo][1]} (fechas NY)")
    print("Cargando velas...")
    arreglos, info = cargar_dias(PERIODOS[periodo][1])

    filas = {}
    for a, r, horas in variantes:
        estrategia = estrategia_de(a)
        carpeta = carpeta_resultados(estrategia, periodo)
        carpeta.mkdir(parents=True, exist_ok=True)
        trades, evaluados = correr(r, horas, arreglos, info, periodo)
        res = resumir(trades, evaluados, r["riesgo"]["capital_inicial_usd"])
        trades.to_csv(carpeta / f"{r['nombre']}_trades.csv", index=False)
        anotar(periodo, r["nombre"], a, huella(r), huella_reglas(r), res)
        guardar_reglas(a, r)
        filas.setdefault(estrategia, []).append(
            {"variante": r["nombre"], "archivo": a.name, "huella": huella(r), **res})
        print(f"  {r['nombre']}: {res['trades']} trades, neto ${res['neto_usd']:,.2f}")

    columnas = ["variante", "trades", "pct_acierto", "profit_factor", "neto_usd",
                "expectativa_R", "max_drawdown_usd", "saldo_final_usd", "retorno_pct",
                "max_drawdown_pct", "racha_perdedora", "sharpe_anual"]
    for estrategia, lista in filas.items():
        carpeta = carpeta_resultados(estrategia, periodo)
        tabla = pd.DataFrame(lista)
        ruta_tabla = carpeta / "comparacion.csv"
        if ruta_tabla.exists():  # conserva las variantes corridas antes; reemplaza las de ahora
            anterior = pd.read_csv(ruta_tabla)
            # quita las variantes cuyo archivo ya no está activo (borrado u oculto con "_")
            activos = {x.name for x in (REGLAS / estrategia).glob("*.toml")
                       if not x.name.startswith("_")}
            anterior = anterior[anterior["archivo"].isin(activos)]
            tabla = pd.concat([anterior[~anterior["variante"].isin(tabla["variante"])], tabla])
        tabla = tabla.sort_values("neto_usd", ascending=False)
        tabla.to_csv(ruta_tabla, index=False)
        print(f"\nComparación {estrategia} ({periodo}), ordenada por neto en USD (con costos):")
        print(tabla[columnas].to_string(index=False))
        print(f"Trades de cada variante en resultados/{estrategia}/{periodo}/<variante>_trades.csv")
    probadas = leer_registro()
    n = probadas.loc[probadas["periodo"] == "in-sample", "huella"].nunique()
    print(f"\nVariantes distintas probadas en in-sample hasta ahora (todas las estrategias): {n}. "
          "Mientras más pruebes, más fácil es que la mejor sea pura suerte.")


if __name__ == "__main__":
    main()
