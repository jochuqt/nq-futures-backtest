"""Paso 2: solo ESTIMA costos y tamaños en Databento. Nunca descarga datos.

Usa solo consultas de metadata (get_dataset_range, get_cost, get_billable_size),
que no consumen crédito. La clave se lee de la variable DATABENTO_API_KEY.
"""
import sys

import databento as db
import pandas as pd

DATASET = "GLBX.MDP3"
SIMBOLO = "NQ.v.0"
STYPE_IN = "continuous"
SCHEMA = "trades"
NY = "America/New_York"
UN_DIA = pd.Timedelta(days=1)


def fin_disponible(cliente):
    """Fecha hasta la que hay datos de trades (exclusiva), en UTC."""
    rango = cliente.metadata.get_dataset_range(dataset=DATASET)
    por_schema = rango.get("schema", {}) or {}
    fin = (por_schema.get(SCHEMA) or {}).get("end") or rango["end"]
    return pd.Timestamp(fin).tz_convert("UTC")


def estimar(cliente, inicio, fin):
    """Devuelve (costo en USD, tamaño en bytes) para el rango [inicio, fin)."""
    params = dict(
        dataset=DATASET,
        symbols=SIMBOLO,
        stype_in=STYPE_IN,
        schema=SCHEMA,
        start=inicio.isoformat(),
        end=fin.isoformat(),
    )
    costo = cliente.metadata.get_cost(**params)
    tamano = cliente.metadata.get_billable_size(**params)
    return costo, tamano


def main():
    try:
        cliente = db.Historical()  # lee la clave de DATABENTO_API_KEY
    except Exception as e:
        sys.exit(f"No se pudo crear el cliente de Databento: {type(e).__name__}")

    fin_datos = fin_disponible(cliente)
    print(f"Hay datos de trades hasta (exclusivo): {fin_datos}")

    # Medianoche UTC del último día completo disponible (fecha final exclusiva).
    limite = fin_datos.normalize()

    # 1 día: el último día hábil (lunes a viernes) completo.
    dia = limite - UN_DIA
    while dia.weekday() >= 5:
        dia -= UN_DIA

    # 1 semana: los últimos 7 días completos.
    semana_ini, semana_fin = limite - 7 * UN_DIA, limite

    # 1 mes: el último mes calendario completo.
    mes_fin = limite.replace(day=1)
    mes_ini = (mes_fin - UN_DIA).replace(day=1)

    filas = []
    filas.append((f"1 día ({dia.date()})", *estimar(cliente, dia, dia + UN_DIA)))
    filas.append(
        (f"1 semana completa ({semana_ini.date()} a {(semana_fin - UN_DIA).date()})",
         *estimar(cliente, semana_ini, semana_fin))
    )

    # Misma semana, pero solo 08:00-17:00 hora de Nueva York, una solicitud por día hábil.
    costo_v, tamano_v, n = 0.0, 0, 0
    d = semana_ini
    while d < semana_fin:
        if d.weekday() < 5:
            ini = pd.Timestamp(d.year, d.month, d.day, 8, tz=NY).tz_convert("UTC")
            fin = pd.Timestamp(d.year, d.month, d.day, 17, tz=NY).tz_convert("UTC")
            c, t = estimar(cliente, ini, fin)
            costo_v += c
            tamano_v += t
            n += 1
        d += UN_DIA
    filas.append((f"1 semana solo 08:00-17:00 NY ({n} solicitudes)", costo_v, tamano_v))

    filas.append(
        (f"1 mes completo ({mes_ini.strftime('%Y-%m')})", *estimar(cliente, mes_ini, mes_fin))
    )

    print()
    print(f"{'Periodo':<55} {'Costo USD':>10} {'Tamaño GB':>10}")
    print("-" * 77)
    for nombre, costo, tamano in filas:
        print(f"{nombre:<55} {costo:>10.2f} {tamano / 1e9:>10.3f}")

    semana_costo, ventana_costo = filas[1][1], filas[2][1]
    if semana_costo > 0:
        print(f"\nLa ventana 08:00-17:00 cuesta el {ventana_costo / semana_costo:.0%} "
              "de la semana completa.")
    print("\n(Tamaño = datos sin comprimir que cobra Databento; el archivo .zst será más chico.)")
    print("Nada fue descargado.")


if __name__ == "__main__":
    main()
