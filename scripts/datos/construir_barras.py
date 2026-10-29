"""Paso 4: construye barras de 1, 5 y 15 minutos desde los trades crudos de data/raw.

Uso:
    python -m scripts.datos.construir_barras              (procesa los meses que falten)
    python -m scripts.datos.construir_barras --reemplazar (rehace las barras ya existentes)

Las barras se arman por MES, juntando todas las partes de ese mes. Cada archivo crudo
declara su rango en el nombre (..._2026-07-01_2026-07-27T0000Z.dbn.zst, en UTC) y de él
solo se usan los trades con ts_recv dentro de ese rango (Databento corta por ts_recv).
Los archivos de data/raw solo se leen, nunca se modifican.
--reemplazar solo afecta a archivos de data/processed.
Las barras se guardan en CSV, con horas ISO que incluyen el desfase (+00:00 en UTC).
No usa la librería databento ni pyarrow (Windows bloquea pyarrow): lee los archivos
con zstandard (descomprimir) + databento_dbn (encabezado) + numpy (registros).
"""
import argparse
import io
import json
from datetime import datetime
from pathlib import Path

import databento_dbn
import numpy as np
import pandas as pd
import zstandard

RAIZ = Path(__file__).resolve().parents[2]
RAW = RAIZ / "data" / "raw"
PROCESSED = RAIZ / "data" / "processed"
NY = "America/New_York"
OTRAS_TEMPORALIDADES = [5, 15]

# Formato binario de un trade de Databento (TradeMsg, 48 bytes, little-endian).
TRADE_DTYPE = np.dtype([
    ("length", "u1"), ("rtype", "u1"), ("publisher_id", "<u2"), ("instrument_id", "<u4"),
    ("ts_event", "<u8"), ("price", "<i8"), ("size", "<u4"), ("action", "S1"),
    ("side", "S1"), ("flags", "u1"), ("depth", "u1"), ("ts_recv", "<u8"),
    ("ts_in_delta", "<i4"), ("sequence", "<u4"),
])
RTYPE_TRADE = 0x00
PRECIO_INDEFINIDO = np.iinfo(np.int64).max  # Databento lo usa para "sin precio"

COLUMNAS = [
    "ts_utc", "ts_ny", "open", "high", "low", "close", "volumen",
    "vol_compra", "vol_venta", "vol_sin_lado", "delta", "n_trades",
    "symbol", "instrument_id",
]
AGREGACION = dict(
    open=("open", "first"),
    high=("high", "max"),
    low=("low", "min"),
    close=("close", "last"),
    volumen=("volumen", "sum"),
    vol_compra=("vol_compra", "sum"),
    vol_venta=("vol_venta", "sum"),
    vol_sin_lado=("vol_sin_lado", "sum"),
    n_trades=("n_trades", "sum"),
    symbol=("symbol", "first"),
)


def leer_fecha(texto):
    """'2026-07-01' o '2026-07-27T0000Z' (UTC) -> Timestamp UTC."""
    formato = "%Y-%m-%dT%H%MZ" if "T" in texto else "%Y-%m-%d"
    return pd.Timestamp(datetime.strptime(texto, formato), tz="UTC")


def rango_del_archivo(ruta):
    """Rango [inicio, fin) en UTC que declara el nombre del archivo crudo."""
    inicio, fin = ruta.name.split(".dbn.zst")[0].split("_")[-2:]
    return leer_fecha(inicio), leer_fecha(fin)


def leer_trades(ruta):
    """Lee un archivo .dbn.zst y devuelve sus trades (dentro del rango de su nombre),
    ordenados por ts_event."""
    inicio_rango, fin_rango = rango_del_archivo(ruta)
    with open(ruta, "rb") as f:
        salida = io.BytesIO()
        zstandard.ZstdDecompressor().copy_stream(f, salida)
    datos = salida.getbuffer()

    # Encabezado: "DBN" + versión (1 byte) + largo de la metadata (4 bytes) + metadata.
    if bytes(datos[:3]) != b"DBN" or datos[3] > 3:
        raise SystemExit(f"{ruta.name}: no parece un archivo DBN versión 1-3.")
    largo_meta = int.from_bytes(datos[4:8], "little")
    inicio = 8 + largo_meta
    decoder = databento_dbn.DBNDecoder()
    decoder.write(bytes(datos[:inicio]))
    metadata = decoder.decode()[0]

    cuerpo = datos[inicio:]
    sobra = len(cuerpo) % TRADE_DTYPE.itemsize
    r = np.frombuffer(cuerpo[: len(cuerpo) - sobra], dtype=TRADE_DTYPE)
    if not ((r["rtype"] == RTYPE_TRADE).all()
            and (r["length"] * 4 == TRADE_DTYPE.itemsize).all()):
        raise SystemExit(f"{ruta.name}: hay registros que no son trades.")

    # Un archivo cortado (trade a medio bajar al final) solo se acepta si el corte
    # quedó DESPUÉS del fin de su rango: así no falta nada de lo que declara.
    recv = r["ts_recv"].astype("int64")
    if sobra and not (len(r) and recv[-1] >= fin_rango.value):
        raise SystemExit(f"{ruta.name}: el archivo está cortado dentro de su rango.")
    dentro = (recv >= inicio_rango.value) & (recv < fin_rango.value)
    if not dentro.all():
        print(f"  {ruta.name}: se dejan fuera {int((~dentro).sum()):,} trades "
              f"fuera de su rango ({inicio_rango} a {fin_rango})")
        r = r[dentro]

    # instrument_id -> símbolo pedido (por ejemplo NQ.v.0), igual que hacía databento.
    simbolos = {int(m["symbol"]): pedido
                for pedido, intervalos in metadata.mappings.items() for m in intervalos}
    ids = pd.Series(r["instrument_id"].astype("int64"))
    sin_simbolo = set(ids.unique()) - set(simbolos)
    if sin_simbolo:
        raise SystemExit(f"{ruta.name}: instrument_id sin símbolo: {sorted(sin_simbolo)}")

    precio = r["price"]
    df = pd.DataFrame({
        "ts_recv": pd.to_datetime(r["ts_recv"].astype("int64"), utc=True),
        "ts_event": pd.to_datetime(r["ts_event"].astype("int64"), utc=True),
        "instrument_id": ids,
        "price": np.where(precio == PRECIO_INDEFINIDO, np.nan, precio / 1e9),  # en dólares
        "size": r["size"].astype("int64"),  # int64 evita errores al restar (delta negativo)
        "side": r["side"].astype("U1"),
        "sequence": r["sequence"].astype("int64"),
        "symbol": ids.map(simbolos),
    })
    return df.sort_values(["ts_event", "sequence"], kind="stable")


def archivos_por_mes():
    """{'2026-07': [archivos de ese mes]}. Comprueba que las partes de cada mes cubran
    el mes completo, sin huecos ni solapes."""
    grupos = {}
    for ruta in sorted(RAW.glob("*.dbn.zst")):
        grupos.setdefault(rango_del_archivo(ruta)[0].strftime("%Y-%m"), []).append(ruta)
    for mes, rutas in grupos.items():
        rutas.sort(key=lambda r: rango_del_archivo(r)[0])
        esperado = pd.Timestamp(f"{mes}-01", tz="UTC")
        for ruta in rutas:
            inicio, fin = rango_del_archivo(ruta)
            if inicio != esperado:
                raise SystemExit(f"{mes}: las partes no calzan ({ruta.name} empieza en "
                                 f"{inicio}, se esperaba {esperado}).")
            esperado = fin
        if esperado != pd.Timestamp(f"{mes}-01", tz="UTC") + pd.offsets.MonthBegin(1):
            raise SystemExit(f"{mes}: las partes no llegan al fin del mes ({esperado}).")
    return grupos


def nombre_mes(mes, rutas):
    """Nombre base de las barras de un mes, p. ej. GLBX.MDP3_trades_NQ.v.0_2026-07."""
    return "_".join(rutas[0].name.split("_")[:-2]) + f"_{mes}"


def leer_trades_mes(rutas):
    """Junta los trades de todas las partes de un mes, ordenados por ts_event."""
    trades = pd.concat([leer_trades(r) for r in rutas], ignore_index=True)
    return trades.sort_values(["ts_event", "sequence"], kind="stable")


def poner_contratos(trades):
    """Cambia 'symbol' (NQ.v.0) por el nombre real del contrato (NQZ5...) según
    data/contratos.json (lo crea obtener_contratos.py). CME reutiliza instrument_id,
    así que se comprueba que cada id se use solo dentro de sus fechas."""
    ruta = RAIZ / "data" / "contratos.json"
    if not ruta.exists():
        raise SystemExit("Falta data/contratos.json: ejecuta scripts\\datos\\obtener_contratos.py")
    contratos = json.loads(ruta.read_text(encoding="utf-8"))
    dia = trades["ts_event"].dt.strftime("%Y-%m-%d")
    for i, n in trades.groupby("instrument_id").size().items():
        c = contratos.get(str(i))
        fuera = 0 if c is None else int(((dia < c["desde"]) | (dia >= c["hasta"]))[
            trades["instrument_id"] == i].sum())
        if c is None or fuera:
            raise SystemExit(f"instrument_id {i}: no está en data/contratos.json o se usa "
                             "fuera de sus fechas; vuelve a ejecutar obtener_contratos.py")
    nombres = {int(i): c["contrato"] for i, c in contratos.items()}
    return trades.assign(symbol=trades["instrument_id"].map(nombres))


def guardar_barras(barras, ruta):
    """Guarda barras en CSV; las horas quedan en ISO con desfase (2026-09-01 13:30:00+00:00)."""
    barras.to_csv(ruta, index=False)


def leer_barras(ruta):
    """Lee un CSV de barras y recupera las horas como fechas con zona horaria."""
    b = pd.read_csv(ruta)
    b["ts_utc"] = pd.to_datetime(b["ts_utc"], utc=True, format="ISO8601")
    b["ts_ny"] = b["ts_utc"].dt.tz_convert(NY)
    return b


def agrupar(tabla, clave):
    """Agrupa filas (trades o barras) en barras según la columna de minuto 'clave'."""
    b = (
        tabla.groupby([clave, "instrument_id"], sort=True)
        .agg(**AGREGACION)
        .reset_index()
        .rename(columns={clave: "ts_utc"})
    )
    b["delta"] = b["vol_compra"] - b["vol_venta"]
    b["ts_ny"] = b["ts_utc"].dt.tz_convert(NY)
    return b[COLUMNAS]


def barras_1m(trades):
    """Barras de 1 minuto. Cada minuto incluye su segundo inicial (09:30:00 a 09:30:59...)."""
    lado = trades["side"]
    tamano = trades["size"]
    tabla = pd.DataFrame({
        "minuto": trades["ts_event"].dt.floor("min"),
        "instrument_id": trades["instrument_id"],
        "open": trades["price"],
        "high": trades["price"],
        "low": trades["price"],
        "close": trades["price"],
        "volumen": tamano,
        "vol_compra": tamano.where(lado == "B", 0),    # B = comprador agresor
        "vol_venta": tamano.where(lado == "A", 0),     # A = vendedor agresor
        "vol_sin_lado": tamano.where(lado == "N", 0),  # N = sin lado
        "n_trades": 1,
        "symbol": trades["symbol"],
    })
    return agrupar(tabla, "minuto")


def remuestrear(barras, minutos):
    """Barras de N minutos construidas desde las de 1 minuto."""
    tabla = barras.assign(bloque=barras["ts_utc"].dt.floor(f"{minutos}min"))
    return agrupar(tabla, "bloque")


def procesar(mes, rutas, reemplazar):
    base = nombre_mes(mes, rutas)
    salidas = {m: PROCESSED / f"barras_{m}m_{base}.csv" for m in [1, *OTRAS_TEMPORALIDADES]}

    existentes = [s for s in salidas.values() if s.exists()]
    if existentes and not reemplazar:
        print(f"Saltado {mes}: ya tiene barras (usa --reemplazar para rehacerlas).")
        return

    print(f"\nMes {mes}: leyendo {', '.join(r.name for r in rutas)} ...")
    trades = poner_contratos(leer_trades_mes(rutas))
    b1 = barras_1m(trades)
    guardar_barras(b1, salidas[1])
    for m in OTRAS_TEMPORALIDADES:
        guardar_barras(remuestrear(b1, m), salidas[m])

    vol_trades = int(trades["size"].sum())
    vol_barras = int(b1["volumen"].sum())
    lados = trades["side"].value_counts()
    print(f"  Trades leídos:        {len(trades):,}")
    print(f"  Barras de 1 minuto:   {len(b1):,}")
    print(f"  Desde (NY):           {b1['ts_ny'].iloc[0]}")
    print(f"  Hasta (NY):           {b1['ts_ny'].iloc[-1]}")
    print(f"  Volumen trades/barras: {vol_trades:,} / {vol_barras:,} "
          f"-> {'coincide' if vol_trades == vol_barras else 'NO COINCIDE'}")
    print(f"  Trades por lado:      B={lados.get('B', 0):,}  A={lados.get('A', 0):,}  "
          f"N={lados.get('N', 0):,} ({lados.get('N', 0) / len(trades):.2%} sin lado)")
    print(f"  Contratos (instrument_id -> symbol): "
          f"{dict(b1.drop_duplicates('instrument_id')[['instrument_id', 'symbol']].values)}")
    for m, s in salidas.items():
        print(f"  Guardado: data/processed/{s.name}")


def main():
    parser = argparse.ArgumentParser(description="Construye barras desde data/raw.")
    parser.add_argument("--reemplazar", action="store_true",
                        help="Rehace barras que ya existen en data/processed")
    args = parser.parse_args()

    grupos = archivos_por_mes()
    if not grupos:
        print("No hay archivos .dbn.zst en data/raw.")
        return
    for mes, rutas in grupos.items():
        procesar(mes, rutas, args.reemplazar)


if __name__ == "__main__":
    main()
