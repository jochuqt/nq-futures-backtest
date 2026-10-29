"""Gráfico de velas interactivo (velas + volumen + delta) en hora de Nueva York.

Uso:
    python -m scripts.informes.grafico_velas --temporalidad 15
    python -m scripts.informes.grafico_velas --temporalidad 1 --desde 2026-09-15 --hasta 2026-09-17

--desde y --hasta son días de NY (--hasta incluido). Crea un archivo .html en
data/processed/graficos que se abre con doble clic en el navegador, sin internet.
"""
import argparse

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.datos.construir_barras import PROCESSED, leer_barras

GRAFICOS = PROCESSED / "graficos"
VERDE, ROJO = "#26a69a", "#ef5350"


def cargar_barras(temporalidad, desde, hasta):
    archivos = sorted(PROCESSED.glob(f"barras_{temporalidad}m_*.csv"))
    if not archivos:
        raise SystemExit(f"No hay barras de {temporalidad} minutos en data/processed.")
    b = pd.concat(leer_barras(a) for a in archivos).sort_values("ts_utc")
    b["hora_ny"] = b["ts_ny"].dt.tz_localize(None)  # el eje muestra la hora de reloj de NY
    if desde:
        b = b[b["hora_ny"] >= pd.Timestamp(desde)]
    if hasta:
        b = b[b["hora_ny"] < pd.Timestamp(hasta) + pd.Timedelta(days=1)]
    if b.empty:
        raise SystemExit("No hay barras en ese rango de fechas.")
    return b.reset_index(drop=True)


def huecos_del_eje(horas, temporalidad):
    """Momentos sin barra (fines de semana, pausa diaria) para ocultarlos del eje."""
    todas = pd.date_range(horas.iloc[0], horas.iloc[-1], freq=f"{temporalidad}min")
    return todas.difference(pd.DatetimeIndex(horas))


def graficar(b, temporalidad):
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.6, 0.2, 0.2],
                        subplot_titles=("", "Volumen", "Delta (compra - venta)"))
    fig.add_trace(go.Candlestick(
        x=b["hora_ny"], open=b["open"], high=b["high"], low=b["low"], close=b["close"],
        increasing_line_color=VERDE, decreasing_line_color=ROJO, name="NQ",
    ), row=1, col=1)
    color_vela = [VERDE if c >= o else ROJO for o, c in zip(b["open"], b["close"])]
    fig.add_trace(go.Bar(x=b["hora_ny"], y=b["volumen"], marker_color=color_vela,
                         name="Volumen"), row=2, col=1)
    color_delta = [VERDE if d >= 0 else ROJO for d in b["delta"]]
    fig.add_trace(go.Bar(x=b["hora_ny"], y=b["delta"], marker_color=color_delta,
                         name="Delta"), row=3, col=1)

    for i in b.index[b["instrument_id"] != b["instrument_id"].shift()][1:]:
        fig.add_vline(x=b.loc[i, "hora_ny"], line_dash="dot", line_color="gray")
        fig.add_annotation(x=b.loc[i, "hora_ny"], y=1, yref="paper",
                           text=f"Roll → {b.loc[i, 'symbol']}",
                           showarrow=False, font_color="gray")

    huecos = huecos_del_eje(b["hora_ny"], temporalidad)
    fig.update_xaxes(rangebreaks=[dict(values=huecos, dvalue=temporalidad * 60_000)])
    fig.update_layout(
        title=f"NQ {temporalidad} minutos · hora Nueva York · "
              f"{b['hora_ny'].iloc[0]:%Y-%m-%d} a {b['hora_ny'].iloc[-1]:%Y-%m-%d}",
        xaxis_rangeslider_visible=False, showlegend=False, template="plotly_dark",
        hovermode="x unified", height=850,
    )
    return fig


def main():
    parser = argparse.ArgumentParser(description="Gráfico de velas de NQ en hora NY.")
    parser.add_argument("--temporalidad", type=int, default=15, choices=[1, 5, 15],
                        help="Minutos por vela")
    parser.add_argument("--desde", help="Primer día NY, formato AAAA-MM-DD")
    parser.add_argument("--hasta", help="Último día NY (incluido), formato AAAA-MM-DD")
    args = parser.parse_args()

    b = cargar_barras(args.temporalidad, args.desde, args.hasta)
    fig = graficar(b, args.temporalidad)

    GRAFICOS.mkdir(exist_ok=True)
    nombre = (f"velas_{args.temporalidad}m_{b['hora_ny'].iloc[0]:%Y-%m-%d}_"
              f"{b['hora_ny'].iloc[-1]:%Y-%m-%d}.html")
    salida = GRAFICOS / nombre
    fig.write_html(salida, include_plotlyjs=True)  # incluye todo: funciona sin internet
    print(f"{len(b):,} velas. Gráfico guardado en data/processed/graficos/{nombre}")


if __name__ == "__main__":
    main()
