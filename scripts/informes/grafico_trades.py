"""Informe gráfico de los trades de una variante, en hora de Nueva York.

Uso:
    python -m scripts.informes.grafico_trades --variante 15M_Breakout_SL11
    python -m scripts.informes.grafico_trades --variante 15M_Breakout_SL11 --periodo completo
    python -m scripts.informes.grafico_trades --variante 15M_Breakout_SL11 --mejores 0 --peores 0  (todos)
    python -m scripts.informes.grafico_trades --estrategia ORB --variante <nombre> --rango-minutos 5

Lee resultados/<estrategia>/<periodo>/<variante>_trades.csv y crea un .html (se abre con doble clic,
sin internet) con: tabla resumen (con costos), tabla por mes, curva de ganancias y un
gráfico por trade (por defecto los 5 mejores y los 5 peores): velas de 5 minutos del día,
rango de apertura sombreado, entrada, stop, objetivo (TP) y salida.
Los periodos out-of-sample y completo solo existen si backtest.py ya los corrió.
"""
import argparse
import html

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from scripts.backtest.backtest import PERIODOS, RESULTADOS, carpeta_resultados
from scripts.datos.construir_barras import PROCESSED, leer_barras

VERDE, ROJO, GRIS, AZUL = "#26a69a", "#ef5350", "#9e9e9e", "#42a5f5"
FONDO = "#111111"
INICIO_DIA, FIN_DIA = "09:00", "16:00"


def cargar_barras(meses, temporalidad):
    partes = []
    for mes in sorted(meses):
        archivos = sorted(PROCESSED.glob(f"barras_{temporalidad}m_*_{mes}.csv"))
        if not archivos:
            raise SystemExit(f"No hay barras de {temporalidad} minutos de {mes} en data/processed.")
        partes += [leer_barras(a) for a in archivos]
    b = pd.concat(partes).sort_values("ts_utc")
    b["hora_ny"] = b["ts_ny"].dt.tz_localize(None)  # el eje muestra la hora de reloj de NY
    return b.reset_index(drop=True)


def curva(t, capital):
    saldo = capital + t["usd_neto"].cumsum()
    fig = go.Figure(go.Scatter(
        x=list(range(1, len(t) + 1)), y=saldo, mode="lines+markers",
        line=dict(color=AZUL, width=2), marker=dict(size=8),
        customdata=t[["fecha", "usd_neto"]],
        hovertemplate="Trade #%{x} · %{customdata[0]}<br>Trade: $%{customdata[1]:,.2f}"
                      "<br>Saldo: $%{y:,.2f}<extra></extra>",
    ))
    fig.add_hline(y=capital, line_color=GRIS, line_width=1,
                  annotation_text=f"Capital inicial ${capital:,.0f}",
                  annotation_font_color=GRIS)
    fig.add_hline(y=0, line_color=ROJO, line_width=1, line_dash="dot",
                  annotation_text="Cuenta en $0", annotation_font_color=ROJO)
    fig.update_layout(title="Saldo de la cuenta (USD neto, con costos)", template="plotly_dark",
                      height=380, xaxis_title="Trade #", yaxis_tickprefix="$",
                      paper_bgcolor=FONDO, plot_bgcolor=FONDO, margin=dict(t=50, b=40))
    return fig


def grafico_trade(n, tr, b, rango_min):
    dia = b[(b["hora_ny"] >= pd.Timestamp(f"{tr.fecha} {INICIO_DIA}"))
            & (b["hora_ny"] <= pd.Timestamp(f"{tr.fecha} {FIN_DIA}"))]
    entrada = pd.Timestamp(f"{tr.fecha} {tr.hora_entrada}")
    salida = pd.Timestamp(f"{tr.fecha} {tr.hora_salida}")
    ini_rango = pd.Timestamp(f"{tr.fecha} 09:30")
    largo = tr.direccion == "largo"

    fig = go.Figure(go.Candlestick(
        x=dia["hora_ny"], open=dia["open"], high=dia["high"], low=dia["low"], close=dia["close"],
        increasing_line_color=VERDE, decreasing_line_color=ROJO, name="NQ 5m",
    ))
    # rango de apertura
    fig.add_shape(type="rect", x0=ini_rango, x1=ini_rango + pd.Timedelta(minutes=rango_min),
                  y0=tr.rango_bajo, y1=tr.rango_alto, fillcolor="rgba(255,255,255,0.08)",
                  line=dict(color=GRIS, width=1, dash="dot"))
    fig.add_annotation(x=ini_rango, y=tr.rango_alto, text="Rango", showarrow=False,
                       xanchor="left", yanchor="bottom", font=dict(color=GRIS, size=11))
    # entrada, stop y objetivo: desde la entrada hasta la salida
    fin_lineas = max(salida, entrada + pd.Timedelta(minutes=15))
    for nivel, color, texto in [(tr.precio_entrada, "#e0e0e0", "Entrada"),
                                (tr.stop_inicial, ROJO, "Stop"),
                                (tr.objetivo, VERDE, "TP")]:
        fig.add_trace(go.Scatter(
            x=[entrada, fin_lineas], y=[nivel, nivel], mode="lines",
            line=dict(color=color, width=2, dash="solid" if texto == "Entrada" else "dash"),
            name=texto, hovertemplate=f"{texto}: %{{y:.2f}}<extra></extra>",
        ))
        fig.add_annotation(x=fin_lineas, y=nivel, text=f"{texto} {nivel:.2f}", showarrow=False,
                           xanchor="left", font=dict(color=color, size=11))
    # marcadores de entrada y salida
    fig.add_trace(go.Scatter(
        x=[entrada], y=[tr.precio_entrada], mode="markers", name="Entrada",
        marker=dict(symbol="triangle-up" if largo else "triangle-down", size=14,
                    color=VERDE if largo else ROJO, line=dict(color="white", width=1)),
        hovertemplate=f"Entrada {tr.direccion} {tr.hora_entrada}: %{{y:.2f}}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[salida], y=[tr.precio_salida], mode="markers", name="Salida",
        marker=dict(symbol="x", size=13, color="#ffca28", line=dict(color="white", width=1)),
        hovertemplate=f"Salida ({tr.motivo_salida}) {tr.hora_salida}: %{{y:.2f}}<extra></extra>",
    ))

    signo = "+" if tr.usd_neto >= 0 else "-"
    fig.update_layout(
        title=(f"#{n} · {tr.fecha} · {tr.contratos} x {tr.contrato} · {tr.direccion}"
               f" · riesgo ${tr.riesgo_usd:,.0f} · salida: {tr.motivo_salida}"
               f" · {tr.R:+.2f} R · {signo}${abs(tr.usd_neto):,.2f}"),
        template="plotly_dark", height=520, showlegend=False, xaxis_rangeslider_visible=False,
        paper_bgcolor=FONDO, plot_bgcolor=FONDO, margin=dict(t=50, b=40, r=110),
        hovermode="closest",
    )
    return fig


def dinero(x):
    clase = "pos" if x > 0 else "neg" if x < 0 else ""
    return f'<span class="{clase}">{"-" if x < 0 else ""}${abs(x):,.2f}</span>'


def tablas(t, s):
    """Tabla resumen (con costos) y tabla por mes, en HTML."""
    curva_r = t["R"].cumsum().to_numpy()
    dd_r = (curva_r - np.maximum.accumulate(np.maximum(curva_r, 0))).min()
    comisiones = t["usd_bruto"].sum() - t["usd_neto"].sum()
    filas = [
        ("Capital inicial", dinero(s.capital_inicial_usd)),
        ("<b>Saldo final</b>", f"<b>{dinero(s.saldo_final_usd)}</b> ({s.retorno_pct:+.1f}%)"),
        ("Saldo mínimo alcanzado", dinero(s.saldo_minimo_usd)),
        ("Riesgo por trade: promedio / máximo",
         f"{dinero(s.riesgo_prom_usd)} / {dinero(s.riesgo_max_usd)}"),
        ("Contratos por trade: promedio / máximo",
         f"{t['contratos'].mean():.1f} / {int(t['contratos'].max())}"),
        ("Trades", f"{int(s.trades)}"),
        ("Días evaluados / con trade", f"{int(s.dias_evaluados)} / {int(s.dias_con_trade)}"),
        ("% de acierto", f"{s.pct_acierto:.1f}%"),
        ("Bruto (con slippage, sin comisiones)", dinero(t["usd_bruto"].sum())),
        ("Comisiones", dinero(-comisiones)),
        ("<b>Neto (con costos)</b>", f"<b>{dinero(s.neto_usd)}</b>"),
        ("Promedio por trade", dinero(s.promedio_usd)),
        ("Ganancia promedio", dinero(s.ganancia_prom_usd)),
        ("Pérdida promedio", dinero(s.perdida_prom_usd)),
        ("Profit factor", f"{s.profit_factor:.2f}"),
        ("Expectativa", f"{s.expectativa_R:+.3f} R"),
        ("Drawdown máximo",
         f"{dinero(s.max_drawdown_usd)} ({s.max_drawdown_pct:.1f}% de la cuenta, {dd_r:.2f} R)"),
        ("Racha perdedora máxima", f"{int(s.racha_perdedora)}"),
        ("Sharpe anual", f"{s.sharpe_anual:.2f}"),
        ("Largos", f"{int(s.largos)} trades, {dinero(s.neto_largos_usd)}"),
        ("Cortos", f"{int(s.cortos)} trades, {dinero(s.neto_cortos_usd)}"),
        ("Salidas: TP / stop / cierre 15:55",
         " / ".join(str(int((t["motivo_salida"] == m).sum()))
                    for m in ("objetivo", "stop", "hora_cierre"))),
    ]
    resumen = "".join(f"<tr><td>{a}</td><td>{b}</td></tr>" for a, b in filas)
    meses = t.assign(mes=t["fecha"].str[:7]).groupby("mes").agg(
        trades=("R", "size"), ganados=("R", lambda x: int((x > 0).sum())),
        neto=("usd_neto", "sum"), R=("R", "sum"))
    por_mes = "".join(
        f"<tr><td>{mes}</td><td>{int(f.trades)}</td><td>{f.ganados}</td>"
        f"<td>{dinero(f.neto)}</td><td>{f.R:+.2f}</td></tr>" for mes, f in meses.iterrows())
    return f"""<h2>Resumen del backtest</h2><div class="tablas">
<table><tr><th>Métrica</th><th>Valor</th></tr>{resumen}</table>
<table><tr><th>Mes</th><th>Trades</th><th>Ganados</th><th>Neto USD</th><th>R</th></tr>{por_mes}</table>
</div>"""


def main():
    parser = argparse.ArgumentParser(description="Informe gráfico de los trades de una variante.")
    parser.add_argument("--variante", required=True, help="nombre de la variante (ej. 15M_Breakout_SL11)")
    parser.add_argument("--estrategia", default="ORB",
                        help="subcarpeta de reglas de la variante (ej. ORB)")
    parser.add_argument("--rango-minutos", type=int, default=15, help="minutos del rango de apertura")
    parser.add_argument("--periodo", default="in-sample", choices=list(PERIODOS))
    parser.add_argument("--mejores", type=int, default=5, help="N mejores trades (USD neto)")
    parser.add_argument("--peores", type=int, default=5, help="N peores trades (USD neto)")
    args = parser.parse_args()

    carpeta = carpeta_resultados(args.estrategia, args.periodo)
    archivo = carpeta / f"{args.variante}_trades.csv"
    if not archivo.exists():
        raise SystemExit(f"No existe {archivo}. Corre primero scripts\\backtest\\backtest.py "
                         f"para ese periodo.")
    t = pd.read_csv(archivo)
    if t.empty:
        raise SystemExit("La variante no tiene trades.")
    desde, hasta = PERIODOS[args.periodo]
    if not t["fecha"].between(desde, hasta).all():
        raise SystemExit(f"Hay trades fuera del periodo {args.periodo}: no se grafican.")
    resumen = pd.read_csv(carpeta / "comparacion.csv").set_index("variante").loc[args.variante]

    t.index = range(1, len(t) + 1)  # número de trade en orden cronológico
    if args.mejores or args.peores:
        orden = t.sort_values("usd_neto", ascending=False)
        grupos = [(f"Los {args.mejores} mejores trades", orden.head(args.mejores)),
                  (f"Los {args.peores} peores trades", orden.tail(args.peores)[::-1])]
        grupos = [(titulo, g) for titulo, g in grupos if len(g)]
        sufijo = f"_mejores{args.mejores}_peores{args.peores}"
    else:
        grupos, sufijo = [("Todos los trades", t)], "_todos"

    elegidos = pd.concat(g for _, g in grupos)
    b = cargar_barras(set(elegidos["fecha"].str[:7]), 5)
    partes = [tablas(t, resumen), "<h2>Rendimiento general</h2>",
              curva(t, resumen.capital_inicial_usd)]
    for titulo, g in grupos:
        partes.append(f"<h2>{html.escape(titulo)}</h2>")
        partes += [grafico_trade(n, tr, b, args.rango_minutos) for n, tr in g.iterrows()]
    cuerpo, plotly_incluido = [], False
    for p in partes:
        if isinstance(p, str):
            cuerpo.append(p)
        else:  # el primer gráfico incluye plotly: funciona sin internet
            cuerpo.append(p.to_html(full_html=False, include_plotlyjs=not plotly_incluido))
            plotly_incluido = True
    cuerpo = "\n".join(cuerpo)
    pagina = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Trades {html.escape(args.variante)}</title>
<style>body{{background:{FONDO};color:#e0e0e0;font-family:Segoe UI,Arial,sans-serif;margin:16px}}
h1{{font-size:20px}} h2{{font-size:18px;margin-top:32px}} p{{color:#bdbdbd}}
.tablas{{display:flex;flex-wrap:wrap;gap:32px;align-items:flex-start}}
table{{border-collapse:collapse;font-size:14px}}
th,td{{padding:4px 12px;border-bottom:1px solid #333;text-align:right}}
th:first-child,td:first-child{{text-align:left}} th{{color:#bdbdbd;font-weight:600}}
.pos{{color:{VERDE}}} .neg{{color:{ROJO}}}</style></head><body>
<h1>{html.escape(args.variante)} · {args.periodo} {desde} a {hasta} · hora Nueva York</h1>
<p>Todos los montos son netos de comisiones y slippage · velas de 5 minutos ·
triángulo = entrada, X amarilla = salida, línea roja = stop, línea verde = TP,
caja punteada = rango de apertura. La curva incluye todos los trades; el # de cada trade
es su número en orden cronológico.</p>
{cuerpo}
</body></html>"""
    destino = carpeta / "graficos"
    destino.mkdir(exist_ok=True)
    salida = destino / f"{args.variante}_trades{sufijo}.html"
    salida.write_text(pagina, encoding="utf-8")
    print(f"{len(elegidos)} trades graficados. Archivo: {salida.relative_to(RESULTADOS.parent)}")


if __name__ == "__main__":
    main()
