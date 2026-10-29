"""Paso 3: descarga un mes, el final de un mes, o un lote de meses de trades de NQ
desde Databento, con controles de gasto.

Uso:
    python -m scripts.datos.descargar --mes 2026-09 --solo-estimar   (nunca descarga)
    python -m scripts.datos.descargar --mes 2026-09                  (pide escribir CONFIRMO)
    python -m scripts.datos.descargar --mes 2026-07 --desde "2026-07-27 10:35" --solo-estimar
        (--desde es hora de Nueva York: baja desde ese minuto hasta el fin del mes)
    python -m scripts.datos.descargar --meses 2026-04 2026-03 2026-02
        (lote: un solo CONFIRMO; baja en ese orden y se detiene en el primer fallo)

Controles:
- Lee config.json y se niega si el costo (en un lote, la SUMA de todos los meses)
  > credito_restante_usd - margen_usd.
- Se niega si hubo una descarga (o un intento) después de la última vez que se guardó
  config.json (para obligar a revisar Billing y actualizar el crédito antes de seguir).
- Nunca sobrescribe archivos en data/raw.
- Anota cada intento en data/INTENTOS.csv ANTES de descargar, y su resultado al final.
- Comprueba que el archivo bajado tenga exactamente los trades que Databento dijo.
- Registra cada descarga completa en data/MANIFEST.csv.
La clave se lee de la variable DATABENTO_API_KEY y nunca se muestra.

No usa la librería databento (Windows bloquea pyarrow, que ella necesita): habla
directo con la API web de Databento usando requests, con las mismas consultas.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests

from scripts.datos.construir_barras import leer_trades

DATASET = "GLBX.MDP3"
SIMBOLO = "NQ.v.0"
STYPE_IN = "continuous"
SCHEMA = "trades"
NY = "America/New_York"

API = "https://hist.databento.com/v0"
ESPERA_SEG = 100  # máximo sin respuesta del servidor (por conexión y por trozo)

RAIZ = Path(__file__).resolve().parents[2]
CONFIG = RAIZ / "config.json"
RAW = RAIZ / "data" / "raw"
MANIFEST = RAIZ / "data" / "MANIFEST.csv"
INTENTOS = RAIZ / "data" / "INTENTOS.csv"
ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def leer_config():
    try:
        datos = json.loads(CONFIG.read_text(encoding="utf-8"))
        credito = float(datos["credito_restante_usd"])
        margen = float(datos["margen_usd"])
    except Exception as e:
        sys.exit(f"config.json no se pudo leer bien ({type(e).__name__}). No se descarga nada.")
    return credito, margen


def filas_csv(ruta, obligatorio):
    if not ruta.exists():
        if obligatorio:
            sys.exit(f"No existe data/{ruta.name}. No se descarga nada.")
        return []
    with ruta.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def anotar_intento(estado, inicio, fin, costo, detalle=""):
    nuevo = not INTENTOS.exists()
    with INTENTOS.open("a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha_hora", "estado", "inicio_utc", "fin_utc",
                        "costo_estimado_usd", "detalle"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), estado,
                    inicio.isoformat(), fin.isoformat(), f"{costo:.4f}", detalle])


def rango(mes, desde):
    try:
        inicio_mes = pd.Timestamp(f"{mes}-01", tz="UTC")
    except Exception:
        sys.exit("El mes debe tener el formato AAAA-MM, por ejemplo 2026-09.")
    fin = inicio_mes + pd.offsets.MonthBegin(1)
    if desde is None:
        return inicio_mes, fin
    try:
        inicio = pd.Timestamp(desde).tz_localize(NY).tz_convert("UTC")
    except Exception:
        sys.exit('--desde debe ser hora de NY, formato "AAAA-MM-DD HH:MM".')
    if not inicio_mes < inicio < fin or inicio != inicio.floor("min"):
        sys.exit("--desde debe caer dentro del mes y en un minuto exacto.")
    return inicio, fin


def nombre_archivo(inicio, fin, desde):
    texto_inicio = inicio.strftime("%Y-%m-%dT%H%MZ") if desde else str(inicio.date())
    return f"{DATASET}_{SCHEMA}_{SIMBOLO}_{texto_inicio}_{fin.date()}.dbn.zst"


class Databento:
    """Consultas a la API web de Databento (las mismas que hace su librería oficial)."""

    def __init__(self):
        clave = os.environ.get("DATABENTO_API_KEY")
        if not clave:
            sys.exit("La variable DATABENTO_API_KEY no está definida. No se descarga nada.")
        self.sesion = requests.Session()
        self.sesion.auth = (clave, "")
        self.sesion.headers["accept"] = "application/json"

    def _revisar(self, r):
        if r.status_code != 200:
            try:
                mensaje = r.json().get("detail", "")
            except Exception:
                mensaje = r.text[:300]
            sys.exit(f"Databento respondió con error {r.status_code}: {mensaje}")

    def _consulta_gratis(self, pedir):
        """Consultas de metadata (gratis): si el servidor falla (5xx), reintenta hasta 3 veces."""
        for intento in range(1, 4):
            r = pedir()
            if r.status_code < 500 or intento == 3:
                break
            print(f"  (Databento respondió {r.status_code}; reintento {intento} de 2...)")
            time.sleep(5)
        self._revisar(r)
        return r.json()

    def get(self, metodo, **params):
        return self._consulta_gratis(
            lambda: self.sesion.get(f"{API}/{metodo}", params=params, timeout=ESPERA_SEG))

    def post(self, metodo, **datos):
        return self._consulta_gratis(
            lambda: self.sesion.post(f"{API}/{metodo}", data=datos, timeout=ESPERA_SEG))

    def descargar(self, ruta, **datos):
        """Baja el archivo trozo a trozo; nunca sobrescribe (modo 'xb'). Cuesta dinero:
        nunca se reintenta sola."""
        with self.sesion.post(f"{API}/timeseries.get_range", data=datos,
                              timeout=ESPERA_SEG, stream=True) as r:
            self._revisar(r)
            with open(ruta, "xb") as f:
                for trozo in r.iter_content(chunk_size=1 << 20):
                    f.write(trozo)


def cotizar(api, mes, desde, fin_datos):
    """Arma el pedido de un mes y lo cotiza (consultas gratis). Devuelve un dict."""
    inicio, fin = rango(mes, desde)
    if fin > fin_datos:
        sys.exit(f"El mes {mes} no está completo todavía (hay datos hasta {fin_datos}). "
                 "No se descarga nada.")
    nombre = nombre_archivo(inicio, fin, desde)
    params = dict(dataset=DATASET, symbols=SIMBOLO, stype_in=STYPE_IN, schema=SCHEMA,
                  start=inicio.isoformat(), end=fin.isoformat())
    return dict(
        mes=mes, desde=desde, inicio=inicio, fin=fin, nombre=nombre, params=params,
        destino=RAW / nombre, parcial=RAW / (nombre + ".parcial"),
        costo=float(api.post("metadata.get_cost", **params, stype_out="instrument_id")),
        tamano=int(api.post("metadata.get_billable_size", **params, stype_out="instrument_id")),
        n_esperado=int(api.post("metadata.get_record_count", **params)),
    )


def bajar(api, p):
    """Descarga y verifica un pedido. Devuelve (trades, bytes en disco) o termina con error."""
    anotar_intento("iniciado", p["inicio"], p["fin"], p["costo"])
    print(f"\nDescargando {p['mes']}...")
    parcial = p["parcial"]
    try:
        api.descargar(parcial, **p["params"], stype_out="instrument_id",
                      encoding="dbn", compression="zstd")
        with parcial.open("rb") as f:
            if f.read(4) != ZSTD_MAGIC:
                raise ValueError("el archivo no parece zstd")
        n_bajado = len(leer_trades(parcial))
        if n_bajado != p["n_esperado"]:
            raise ValueError(f"tiene {n_bajado:,} trades y Databento dijo {p['n_esperado']:,}")
    except BaseException as e:  # también si se corta con Ctrl+C
        anotar_intento("fallido", p["inicio"], p["fin"], p["costo"], f"{type(e).__name__}: {e}")
        print(f"\nLa descarga de {p['mes']} NO quedó completa ({type(e).__name__}: {e}).")
        if parcial.exists():
            print(f"Lo bajado quedó como data/raw/{parcial.name} para revisarlo.")
        print("No se bajan más meses. Revisa Billing en Databento: puede haber cobro "
              "aunque se cortara.")
        sys.exit(1)

    parcial.rename(p["destino"])  # en Windows falla si el destino ya existe: nunca sobrescribe
    tamano_disco = p["destino"].stat().st_size
    anotar_intento("ok", p["inicio"], p["fin"], p["costo"], f"{n_bajado} trades")

    with MANIFEST.open("a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            datetime.now().isoformat(timespec="seconds"), DATASET, SCHEMA, SIMBOLO,
            p["inicio"].isoformat() if p["desde"] else p["inicio"].date().isoformat(),
            p["fin"].date().isoformat(), f"{p['costo']:.4f}", tamano_disco, p["nombre"],
        ])
    print(f"Listo: data/raw/{p['nombre']} ({tamano_disco / 1e9:.3f} GB en disco)")
    print(f"Trades: {n_bajado:,} (coincide con lo que dijo Databento)")


def main():
    parser = argparse.ArgumentParser(description="Descarga meses de trades de NQ.")
    cuales = parser.add_mutually_exclusive_group(required=True)
    cuales.add_argument("--mes", help="Mes a descargar, formato AAAA-MM")
    cuales.add_argument("--meses", nargs="+", help="Lote de meses AAAA-MM, en orden de descarga")
    parser.add_argument("--desde", help='Hora de NY desde la que bajar, "AAAA-MM-DD HH:MM" '
                                        "(solo con --mes)")
    parser.add_argument("--solo-estimar", action="store_true",
                        help="Solo muestra costo, tamaño y conteo; nunca descarga")
    args = parser.parse_args()
    if args.desde and args.meses:
        sys.exit("--desde solo se puede usar con --mes (un solo mes).")
    meses = args.meses or [args.mes]
    if len(set(meses)) != len(meses):
        sys.exit("Hay meses repetidos en el lote. No se descarga nada.")

    credito, margen = leer_config()
    disponible = credito - margen
    manifest = filas_csv(MANIFEST, obligatorio=True)
    intentos = filas_csv(INTENTOS, obligatorio=False)

    api = Databento()
    rango_datos = api.get("metadata.get_dataset_range", dataset=DATASET)
    fin_datos = pd.Timestamp(
        ((rango_datos.get("schema") or {}).get(SCHEMA) or {}).get("end") or rango_datos["end"]
    ).tz_convert("UTC")

    pedidos = [cotizar(api, m, args.desde, fin_datos) for m in meses]
    costo_total = sum(p["costo"] for p in pedidos)
    acumulado = sum(float(r["costo_estimado_usd"]) for r in manifest)

    print(f"Dataset:          {DATASET}")
    print(f"Símbolo:          {SIMBOLO} (stype_in={STYPE_IN})")
    print(f"Schema:           {SCHEMA}")
    for p in pedidos:
        print(f"\n  Mes {p['mes']}")
        print(f"  Rango (UTC):      {p['inicio']} hasta {p['fin']} (final exclusivo)")
        print(f"  Rango (NY):       {p['inicio'].tz_convert(NY)} hasta {p['fin'].tz_convert(NY)}")
        print(f"  Costo estimado:   ${p['costo']:.2f}")
        print(f"  Tamaño:           {p['tamano'] / 1e9:.3f} GB (sin comprimir)")
        print(f"  Trades esperados: {p['n_esperado']:,}")
        print(f"  Archivo destino:  data/raw/{p['nombre']}")
    print(f"\nCosto estimado TOTAL: ${costo_total:.2f} ({len(pedidos)} mes(es))")
    print(f"Crédito en config: ${credito:.2f} - margen ${margen:.2f} = ${disponible:.2f} disponibles")
    print(f"Ya descargado según MANIFEST: ${acumulado:.2f} estimados")

    problemas = []
    if costo_total > disponible:
        problemas.append(f"el costo total (${costo_total:.2f}) supera lo disponible "
                         f"(${disponible:.2f})")
    for p in pedidos:
        if p["destino"].exists() or p["parcial"].exists():
            problemas.append(f"ya existe un archivo para {p['mes']} en data/raw "
                             "(no se sobrescribe)")
    fechas = [datetime.fromisoformat(r["fecha_hora"]) for r in manifest + intentos]
    if fechas:
        guardado = datetime.fromtimestamp(CONFIG.stat().st_mtime)
        if guardado <= max(fechas):
            problemas.append("config.json no se actualizó después de la última descarga "
                             "o intento (revisa Billing y guarda el crédito nuevo)")

    if problemas:
        print("\nNO se puede descargar porque:")
        for p in problemas:
            print(f"  - {p}")
        sys.exit(1)

    if args.solo_estimar:
        print("\nModo --solo-estimar: los controles están OK, pero no se descarga nada.")
        return

    respuesta = input(f"\nEscribe CONFIRMO para descargar {len(pedidos)} mes(es) por "
                      f"${costo_total:.2f} (cualquier otra cosa cancela): ")
    if respuesta.strip() != "CONFIRMO":
        print("Cancelado. No se descargó nada.")
        return

    for p in pedidos:
        bajar(api, p)

    print(f"\nTodo listo: {len(pedidos)} mes(es) descargados y verificados.")
    print(f"Costo estimado acumulado según MANIFEST: ${acumulado + costo_total:.2f}")
    print("Revisa Billing en Databento y actualiza credito_restante_usd en config.json "
          "antes de la próxima descarga.")


if __name__ == "__main__":
    main()
