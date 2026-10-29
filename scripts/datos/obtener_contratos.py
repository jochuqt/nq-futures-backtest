"""Averigua el nombre real de cada contrato (NQZ5, NQH6...) y lo guarda en data/contratos.json.

Uso:
    python -m scripts.datos.obtener_contratos

Lee los instrument_id del encabezado de cada archivo de data/raw y le pregunta a
Databento su nombre con symbology.resolve (consulta GRATUITA, no descarga datos).
construir_barras.py usa data/contratos.json para poner el nombre en las barras.
"""
import json
import sys

import databento_dbn
import zstandard

from scripts.datos.construir_barras import RAIZ, RAW
from scripts.datos.descargar import DATASET, Databento

CONTRATOS = RAIZ / "data" / "contratos.json"


def leer_metadata(ruta):
    """Lee solo el encabezado (metadata) de un archivo .dbn.zst."""
    with open(ruta, "rb") as f, zstandard.ZstdDecompressor().stream_reader(f) as lector:
        inicio = lector.read(8)
        largo = int.from_bytes(inicio[4:8], "little")
        resto = b""
        while len(resto) < largo:
            trozo = lector.read(largo - len(resto))
            if not trozo:
                break
            resto += trozo
    decoder = databento_dbn.DBNDecoder()
    decoder.write(inicio + resto)
    return decoder.decode()[0]


def main():
    archivos = sorted(RAW.glob("*.dbn.zst"))
    if not archivos:
        sys.exit("No hay archivos .dbn.zst en data/raw.")

    # instrument_id -> primer y último día (UTC) en que aparece en los archivos
    dias = {}
    for ruta in archivos:
        for intervalos in leer_metadata(ruta).mappings.values():
            for m in intervalos:
                i = int(m["symbol"])
                d0, d1 = str(m["start_date"]), str(m["end_date"])
                antes = dias.get(i, (d0, d1))
                dias[i] = (min(antes[0], d0), max(antes[1], d1))

    inicio = min(d0 for d0, _ in dias.values())
    fin = max(d1 for _, d1 in dias.values())
    respuesta = Databento().post(
        "symbology.resolve", dataset=DATASET, symbols=",".join(map(str, sorted(dias))),
        stype_in="instrument_id", stype_out="raw_symbol", start_date=inicio, end_date=fin,
    )
    if respuesta.get("not_found"):
        sys.exit(f"Databento no encontró estos instrument_id: {respuesta['not_found']}")

    # CME reutiliza instrument_id con el tiempo: solo cuentan los nombres vigentes en
    # los días en que ese id aparece en nuestros archivos.
    contratos = {}
    for i in sorted(dias):
        d0, d1 = dias[i]
        nombres = {m["s"] for m in respuesta["result"].get(str(i), [])
                   if m["d0"] < d1 and m["d1"] > d0}
        if len(nombres) != 1:
            sys.exit(f"instrument_id {i}: se esperaba un solo nombre y llegó {sorted(nombres)}")
        contratos[str(i)] = {"contrato": nombres.pop(), "desde": d0, "hasta": d1}

    CONTRATOS.write_text(json.dumps(contratos, indent=2) + "\n", encoding="utf-8")
    print("instrument_id -> contrato (días UTC en que aparece en los archivos; "
          "'hasta' es exclusivo):")
    for i in sorted(dias, key=lambda i: dias[i]):
        print(f"  {i:>10} -> {contratos[str(i)]['contrato']:6}  ({dias[i][0]} a {dias[i][1]})")
    print(f"\nGuardado en data/{CONTRATOS.name}")


if __name__ == "__main__":
    main()
