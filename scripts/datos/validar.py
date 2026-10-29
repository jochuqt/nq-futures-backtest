"""Paso 5: valida los trades crudos y las barras de 1 minuto. Todo en hora de Nueva York.

Uso:
    python -m scripts.datos.validar

Solo lee archivos. Escribe el reporte en data/processed/reporte_validacion.txt.
"""
import pandas as pd
from pandas.tseries.holiday import (
    AbstractHolidayCalendar, GoodFriday, Holiday, USLaborDay, USMartinLutherKingJr,
    USMemorialDay, USPresidentsDay, USThanksgivingDay, nearest_workday, sunday_to_monday,
)

from scripts.datos.construir_barras import (PROCESSED, archivos_por_mes, leer_barras, leer_trades_mes,
                                            nombre_mes)

INICIO_SESION = pd.Timedelta(hours=18)  # la sesión de CME empieza 18:00 NY del día anterior
FIN_SESION = pd.Timedelta(hours=17)
APERTURA = pd.Timedelta(hours=9, minutes=30)
CIERRE_RTH = pd.Timedelta(hours=16)
UN_MIN = pd.Timedelta(minutes=1)
UTC_NS = "datetime64[ns, UTC]"


class CalendarioNYSE(AbstractHolidayCalendar):
    """Feriados habituales de la bolsa de Nueva York (no incluye cierres especiales)."""
    rules = [
        Holiday("Año Nuevo", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independencia", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Navidad", month=12, day=25, observance=nearest_workday),
    ]


reporte = []


def decir(texto=""):
    print(texto)
    reporte.append(texto)


def fecha_sesion(ts_ny):
    """Día de sesión en hora NY: lo que pasa desde las 18:00 cuenta para el día siguiente."""
    return (ts_ny.dt.tz_localize(None) + (pd.Timedelta(days=1) - INICIO_SESION)).dt.normalize()


def revisar_mes(mes, rutas, problemas):
    """Revisa los trades crudos de un mes contra sus barras de 1 minuto. Devuelve las barras."""
    ruta_barras = PROCESSED / f"barras_1m_{nombre_mes(mes, rutas)}.csv"
    if not ruta_barras.exists():
        problemas.append(f"Faltan las barras de {mes} (ejecuta construir_barras.py).")
        return None

    decir(f"\n== {mes} ({', '.join(r.name for r in rutas)})")
    barras = leer_barras(ruta_barras)
    trades = leer_trades_mes(rutas)
    decir(f"  Trades del mes: {len(trades):,}")

    precio = trades["price"]
    malos = int((precio.isna() | (precio <= 0)).sum())
    decir(f"  Trades con precio inválido (vacío, cero o negativo): {malos:,}")
    if malos:
        problemas.append(f"{malos:,} trades con precio inválido en {mes}.")

    n_lado_n = int((trades["side"] == "N").sum())
    decir(f"  Trades sin lado (N): {n_lado_n:,} de {len(trades):,} "
          f"({n_lado_n / len(trades):.4%})")

    por_minuto = (
        trades.assign(ts_utc=trades["ts_event"].dt.floor("min").astype(UTC_NS))
        .groupby(["ts_utc", "instrument_id"])
        .agg(vol_trades=("size", "sum"), n_trades_crudo=("size", "size"))
        .reset_index()
    )
    comparacion = barras.assign(ts_utc=barras["ts_utc"].astype(UTC_NS)).merge(
        por_minuto, on=["ts_utc", "instrument_id"], how="outer", indicator=True
    )
    distintos = int((
        (comparacion["_merge"] != "both")
        | (comparacion["volumen"] != comparacion["vol_trades"])
        | (comparacion["n_trades"] != comparacion["n_trades_crudo"])
    ).sum())
    decir(f"  Minutos donde volumen o n_trades no coincide con los trades: {distintos:,}")
    if distintos:
        problemas.append(f"{distintos:,} minutos con volumen distinto al de los trades "
                         f"en {mes}.")
    return barras


def main():
    problemas = []
    calendario = CalendarioNYSE()

    decir("REPORTE DE VALIDACIÓN (todas las horas en Nueva York)")
    lista = [revisar_mes(m, rutas, problemas) for m, rutas in archivos_por_mes().items()]
    lista = [b for b in lista if b is not None]
    if not lista:
        decir("No hay barras para validar.")
        return
    b = pd.concat(lista).sort_values(["ts_utc", "instrument_id"]).reset_index(drop=True)
    hora_ny = b["ts_ny"].dt.tz_localize(None)  # hora "de reloj" en Nueva York
    b["sesion"] = fecha_sesion(b["ts_ny"])

    # --- Duplicados ---
    decir("\n== Duplicados")
    dup_exactos = int(b.duplicated(["ts_utc", "instrument_id"]).sum())
    dup_minuto = int(b.duplicated("ts_utc").sum())
    decir(f"  Barras repetidas (mismo minuto y contrato): {dup_exactos:,}")
    decir(f"  Minutos con más de un contrato a la vez: {dup_minuto - dup_exactos:,}")
    if dup_exactos:
        problemas.append(f"{dup_exactos:,} barras repetidas.")
    if dup_minuto - dup_exactos:
        problemas.append(f"{dup_minuto - dup_exactos:,} minutos con dos contratos a la vez.")

    # --- Precios inválidos ---
    decir("\n== Precios inválidos en barras")
    ohlc = b[["open", "high", "low", "close"]]
    vacios_o_no_positivos = int((ohlc.isna() | (ohlc <= 0)).any(axis=1).sum())
    high_menor_low = int((b["high"] < b["low"]).sum())
    fuera_rango = int((
        (b["open"] > b["high"]) | (b["open"] < b["low"])
        | (b["close"] > b["high"]) | (b["close"] < b["low"])
    ).sum())
    decir(f"  Barras con precio vacío, cero o negativo: {vacios_o_no_positivos:,}")
    decir(f"  Barras con high < low: {high_menor_low:,}")
    decir(f"  Barras con open o close fuera del rango high-low: {fuera_rango:,}")
    if vacios_o_no_positivos + high_menor_low + fuera_rango:
        problemas.append("Hay barras con precios inválidos (ver sección de precios).")

    # --- Sesiones por año ---
    inicio_datos, fin_datos = hora_ny.min(), hora_ny.max() + UN_MIN
    sesiones = pd.Series(sorted(b["sesion"].unique()))
    completa = ((sesiones - pd.Timedelta(days=1) + INICIO_SESION >= inicio_datos)
                & (sesiones + FIN_SESION <= fin_datos))
    parciales = sesiones[~completa]
    completas = sesiones[completa]
    decir(f"\n== Sesiones (una sesión va de 18:00 NY del día anterior a 17:00 NY)")
    decir(f"  Datos desde {inicio_datos} hasta {fin_datos} (hora NY)")
    decir(f"  Sesiones cortadas por el inicio o fin de los datos: "
          f"{', '.join(str(s.date()) for s in parciales) or 'ninguna'}")

    fines_de_semana = sesiones[sesiones.dt.weekday >= 5]
    if len(fines_de_semana):
        problemas.append(f"Hay sesiones en fin de semana: "
                         f"{', '.join(str(s.date()) for s in fines_de_semana)}.")

    feriados = pd.Series(dtype=object)
    if len(completas):
        feriados = calendario.holidays(completas.min(), completas.max(), return_name=True)
        habiles = pd.bdate_range(completas.min(), completas.max())  # lunes a viernes
        con_datos = set(completas)
        decir(f"\n  {'Año':<6}{'Días L-V':>9}{'Feriados NYSE':>15}{'Sesiones con datos':>20}")
        for anio in sorted(set(habiles.year)):
            del_anio = habiles[habiles.year == anio]
            fer = sum(d in feriados.index for d in del_anio)
            ses = sum(d in con_datos for d in del_anio)
            decir(f"  {anio:<6}{len(del_anio):>9}{fer:>15}{ses:>20}")
        faltan = [d for d in habiles if d not in con_datos]
        for d in faltan:
            nombre = feriados.get(d)
            motivo = f"feriado NYSE: {nombre}" if nombre else "NO es feriado conocido"
            decir(f"  Día L-V sin datos: {d.date()} ({motivo})")
            if not nombre:
                problemas.append(f"Falta la sesión del {d.date()} y no es feriado conocido.")
        if feriados.size:
            decir("  Feriados NYSE en el periodo (CME suele abrir con horario reducido): "
                  + ", ".join(f"{d.date()} {n}" for d, n in feriados.items()))

    # --- Vela de las 09:30 NY ---
    decir("\n== Vela de las 09:30 NY en días regulares (L-V que no son feriado NYSE)")
    regulares = [s for s in completas if s.weekday() < 5 and s not in feriados.index]
    minutos_existentes = set(hora_ny)
    sin_0930 = [s for s in regulares if s + APERTURA not in minutos_existentes]
    decir(f"  Días regulares: {len(regulares)}   Sin vela de 09:30: {len(sin_0930)}")
    for s in sin_0930:
        decir(f"    Falta 09:30 del {s.date()}")
    if sin_0930:
        problemas.append(f"{len(sin_0930)} días regulares sin vela de las 09:30.")

    # --- Huecos dentro de cada sesión ---
    decir("\n== Huecos (minutos sin ningún trade dentro de una misma sesión)")
    salto = b["ts_utc"].diff()
    misma_sesion = b["sesion"] == b["sesion"].shift()
    es_hueco = misma_sesion & (salto > UN_MIN)
    huecos = b[es_hueco].assign(
        faltan=(salto[es_hueco] / UN_MIN - 1).astype(int),
        desde=(hora_ny.shift() + UN_MIN)[es_hueco],
    )
    decir(f"  Huecos: {len(huecos):,}   Minutos sin trades en total: {huecos['faltan'].sum():,}")
    for _, h in huecos.nlargest(5, "faltan").iterrows():
        decir(f"    {h['faltan']:>4} min sin trades desde {h['desde']}")
    regulares_set = set(regulares)
    hora_del_dia = huecos["desde"] - huecos["desde"].dt.normalize()
    en_rth = huecos[
        huecos["sesion"].isin(regulares_set)
        & (hora_del_dia >= APERTURA) & (hora_del_dia < CIERRE_RTH)
    ]
    decir(f"  Huecos en horario regular (09:30-16:00) de días regulares: {len(en_rth):,}")
    for _, h in en_rth.head(10).iterrows():
        decir(f"    {h['faltan']:>4} min sin trades desde {h['desde']}")
    if len(en_rth):
        problemas.append(f"{len(en_rth):,} huecos dentro del horario regular 09:30-16:00.")

    # --- Roll (cambio de contrato) ---
    decir("\n== Roll (cambio de instrument_id)")
    cambio = b["instrument_id"] != b["instrument_id"].shift()
    cambio.iloc[0] = False
    for i in b.index[cambio]:
        antes, despues = b.loc[i - 1], b.loc[i]
        decir(f"  {hora_ny[i]} (sesión {despues['sesion'].date()}): "
              f"{antes['symbol']} ({antes['instrument_id']}) -> "
              f"{despues['symbol']} ({despues['instrument_id']})  "
              f"cierre previo {antes['close']:.2f}, apertura nueva {despues['open']:.2f}, "
              f"salto {despues['open'] - antes['close']:+.2f} puntos (precios sin ajustar)")
    if not cambio.any():
        decir("  Sin cambios de contrato en el periodo.")

    # --- Resumen ---
    decir("\n== RESUMEN")
    if problemas:
        for p in problemas:
            decir(f"  - {p}")
    else:
        decir("  No se encontraron problemas.")

    salida = PROCESSED / "reporte_validacion.txt"
    salida.write_text("\n".join(reporte) + "\n", encoding="utf-8")
    print(f"\nReporte guardado en data/processed/{salida.name}")


if __name__ == "__main__":
    main()
