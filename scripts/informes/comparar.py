"""Informe comparativo de varias variantes en un mismo periodo, en hora de Nueva York.

Uso:
    python -m scripts.informes.comparar --periodo completo                (todas las ORB del periodo)
    python -m scripts.informes.comparar --periodo completo --variantes ORB_A ORB_B ORB_C
    python -m scripts.informes.comparar --estrategia ORB --rango-minutos 5

Lee resultados/<estrategia>/<periodo>/comparacion.csv y los trades de cada variante (backtest.py debe
haberlas corrido en ese periodo) y crea un .html (se abre con doble clic, sin internet) con:
tabla comparativa (con costos), un gráfico con el saldo de la cuenta de cada variante, sus
reglas lado a lado (de resultados/reglas_probadas) y, por variante, su mejor y su peor trade.
"""
import argparse
import html

import pandas as pd
import plotly.graph_objects as go

from scripts.backtest.backtest import PERIODOS, RESULTADOS, carpeta_resultados, reglas_probadas
from scripts.informes.grafico_trades import FONDO, GRIS, ROJO, cargar_barras, dinero, grafico_trade

# colores por variante (orden fijo) y estilos de línea para distinguirlas aunque se superpongan
COLORES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#9085e9"]
LINEAS = ["solid", "dash", "dot", "dashdot", "longdash", "longdashdot"]


def curvas(variantes, trades, capital, desde):
    fig = go.Figure()
    abajo, arriba, vistas = [], [], {}
    for k, v in enumerate(variantes):
        t = trades[v]
        x = [pd.Timestamp(desde)] + list(pd.to_datetime(t["fecha"] + " " + t["hora_salida"]))
        y = [capital[v]] + list(capital[v] + t["usd_neto"].cumsum())
        clave = (tuple(x), tuple(y))
        igual = vistas.setdefault(clave, v)  # ¿otra variante ya tiene exactamente esta curva?
        repetida = igual != v
        traza = go.Scatter(
            x=x, y=y, mode="lines", name=f"{v} (= {igual})" if repetida else v,
            opacity=0.45 if repetida else 1,
            line=dict(color=COLORES[k % len(COLORES)], width=8 if repetida else 2,
                      dash="solid" if repetida else LINEAS[k % len(LINEAS)], shape="hv"),
            hovertemplate=f"{html.escape(v)}<br>%{{x|%Y-%m-%d %H:%M}}<br>Saldo: $%{{y:,.2f}}"
                          "<extra></extra>",
        )
        # las curvas idénticas a otra van gruesas y por debajo, para que se vean las dos
        (abajo if repetida else arriba).append(traza)
    for traza in abajo + arriba:
        fig.add_trace(traza)
    base = list(capital.values())[0]
    fig.add_hline(y=base, line_color=GRIS, line_width=1,
                  annotation_text=f"Capital inicial ${base:,.0f}", annotation_font_color=GRIS)
    fig.add_hline(y=0, line_color=ROJO, line_width=1, line_dash="dot",
                  annotation_text="Cuenta en $0", annotation_font_color=ROJO)
    fig.update_layout(title="Saldo de la cuenta de cada estrategia (USD neto, con costos)",
                      template="plotly_dark", height=520, yaxis_tickprefix="$",
                      paper_bgcolor=FONDO, plot_bgcolor=FONDO, hovermode="closest",
                      legend=dict(orientation="h", y=-0.15), margin=dict(t=50, b=40))
    return fig


def tabla(variantes, res, trades):
    def fila(titulo, f):
        return f"<tr><td>{titulo}</td>" + "".join(f"<td>{f(v)}</td>" for v in variantes) + "</tr>"

    s = {v: res.loc[v] for v in variantes}
    quemada = {v: s[v].saldo_minimo_usd <= 0 for v in variantes}
    filas = [
        fila("Trades", lambda v: int(s[v].trades)),
        fila("% de acierto", lambda v: f"{s[v].pct_acierto:.1f}%"),
        fila("<b>Neto (con costos)</b>", lambda v: f"<b>{dinero(s[v].neto_usd)}</b>"),
        fila("<b>Saldo final</b>",
             lambda v: f"<b>{dinero(s[v].saldo_final_usd)}</b> ({s[v].retorno_pct:+.1f}%)"),
        fila("Saldo mínimo", lambda v: dinero(s[v].saldo_minimo_usd)),
        fila("¿La cuenta llegó a $0?",
             lambda v: '<span class="neg">SÍ</span>' if quemada[v] else "no"),
        fila("Profit factor", lambda v: f"{s[v].profit_factor:.2f}"),
        fila("Expectativa", lambda v: f"{s[v].expectativa_R:+.3f} R"),
        fila("Drawdown máximo",
             lambda v: f"{dinero(s[v].max_drawdown_usd)} ({s[v].max_drawdown_pct:.1f}%)"),
        fila("Racha perdedora máxima", lambda v: int(s[v].racha_perdedora)),
        fila("Sharpe anual", lambda v: f"{s[v].sharpe_anual:.2f}"),
        fila("Ganancia / pérdida promedio",
             lambda v: f"{dinero(s[v].ganancia_prom_usd)} / {dinero(s[v].perdida_prom_usd)}"),
        fila("Riesgo por trade: promedio / máximo",
             lambda v: f"{dinero(s[v].riesgo_prom_usd)} / {dinero(s[v].riesgo_max_usd)}"),
        fila("Largos", lambda v: f"{int(s[v].largos)} · {dinero(s[v].neto_largos_usd)}"),
        fila("Cortos", lambda v: f"{int(s[v].cortos)} · {dinero(s[v].neto_cortos_usd)}"),
        fila("Salidas: TP / stop / cierre 15:55", lambda v: " / ".join(
            str(int((trades[v]["motivo_salida"] == m).sum()))
            for m in ("objetivo", "stop", "hora_cierre"))),
    ]
    cabecera = "<tr><th>Métrica</th>" + "".join(f"<th>{html.escape(v)}</th>" for v in variantes)
    return f"<table>{cabecera}</tr>{''.join(filas)}</table>"


def aplanar(reglas):
    """{"stop": {"tipo": ...}} -> {"stop.tipo": ...} (sin nombre ni descripción)."""
    plano = {}
    for seccion, valores in reglas.items():
        if isinstance(valores, dict):
            plano.update({f"{seccion}.{k}": v for k, v in valores.items()})
    return plano


def tabla_reglas(variantes, res):
    """Parámetros de cada variante lado a lado: primero los que cambian entre ellas."""
    planos = {}
    for v in variantes:
        reglas = reglas_probadas(v, res.loc[v, "huella"])
        planos[v] = aplanar(reglas) if reglas else None
    sin_copia = [v for v in variantes if planos[v] is None]
    con_copia = [v for v in variantes if planos[v] is not None]
    aviso = (f"<p>Sin copia de reglas (se corrieron antes de que existiera esta función; "
             f"vuelve a correrlas con backtest.py): {html.escape(', '.join(sin_copia))}</p>"
             if sin_copia else "")
    if not con_copia:
        return aviso
    claves = list(planos[con_copia[0]])
    distintas = [k for k in claves if len({repr(planos[v].get(k)) for v in con_copia}) > 1]
    iguales = [k for k in claves if k not in distintas]

    def texto(x):
        if isinstance(x, bool):
            return "sí" if x else "no"
        if isinstance(x, list):
            return ", ".join(map(str, x)) or "(ninguna)"
        return str(x)

    def filas(lista):
        return "".join(f"<tr><td>{k}</td>" + "".join(
            f"<td>{html.escape(texto(planos[v].get(k)))}</td>" for v in con_copia) + "</tr>"
            for k in lista)

    cabecera = "<tr><th>Parámetro</th>" + "".join(f"<th>{html.escape(v)}</th>" for v in con_copia)
    if distintas:
        cambian = f"<p>Parámetros que cambian entre estrategias:</p><table>{cabecera}</tr>" \
                  f"{filas(distintas)}</table>"
    else:
        cambian = "<p>Todas tienen exactamente los mismos parámetros.</p>"
    return (aviso + cambian + f"<details><summary>Ver los {len(iguales)} parámetros iguales en "
            f"todas</summary><table>{cabecera}</tr>{filas(iguales)}</table></details>")


def main():
    parser = argparse.ArgumentParser(description="Informe comparativo de variantes.")
    parser.add_argument("--periodo", default="in-sample", choices=list(PERIODOS))
    parser.add_argument("--estrategia", default="ORB",
                        help="subcarpeta de reglas (ej. ORB)")
    parser.add_argument("--variantes", nargs="+",
                        help="nombres de las variantes (por defecto: todas las del periodo)")
    parser.add_argument("--rango-minutos", type=int, default=15, help="minutos del rango de apertura")
    args = parser.parse_args()

    carpeta = carpeta_resultados(args.estrategia, args.periodo)
    res = pd.read_csv(carpeta / "comparacion.csv", dtype={"huella": str}).set_index("variante")
    if not args.variantes:  # ordenadas por saldo final, la mejor primero
        args.variantes = list(res.sort_values("saldo_final_usd", ascending=False).index)
    faltan = [v for v in args.variantes if v not in res.index]
    if faltan:
        raise SystemExit(f"Sin resultados en {args.periodo} para: {', '.join(faltan)}. "
                         "Corre primero scripts\\backtest\\backtest.py para ese periodo.")
    desde, hasta = PERIODOS[args.periodo]
    trades = {}
    for v in args.variantes:
        t = pd.read_csv(carpeta / f"{v}_trades.csv")
        if not t["fecha"].between(desde, hasta).all():
            raise SystemExit(f"{v}: hay trades fuera del periodo {args.periodo}.")
        t.index = range(1, len(t) + 1)  # número de trade en orden cronológico
        trades[v] = t
    capital = {v: res.loc[v, "capital_inicial_usd"] for v in args.variantes}

    elegidos = {v: [("Mejor trade", t.loc[t["usd_neto"].idxmax()]),
                    ("Peor trade", t.loc[t["usd_neto"].idxmin()])]
                for v, t in trades.items() if len(t)}
    meses = {tr["fecha"][:7] for lista in elegidos.values() for _, tr in lista}
    b = cargar_barras(meses, 5) if meses else None

    partes = ["<h2>Tabla comparativa</h2>", tabla(args.variantes, res, trades),
              "<h2>Rendimiento</h2>", curvas(args.variantes, trades, capital, desde),
              "<h2>Reglas de cada estrategia</h2>", tabla_reglas(args.variantes, res)]
    for v, lista in elegidos.items():
        partes.append(f"<h2>{html.escape(v)}</h2><p>{html.escape(res.loc[v, 'archivo'])}</p>")
        for titulo, tr in lista:
            fig = grafico_trade(tr.name, tr, b, args.rango_minutos)
            fig.update_layout(title=f"{titulo} · {fig.layout.title.text}")
            partes.append(fig)

    cuerpo, plotly_incluido = [], False
    for p in partes:
        if isinstance(p, str):
            cuerpo.append(p)
        else:  # el primer gráfico incluye plotly: funciona sin internet
            cuerpo.append(p.to_html(full_html=False, include_plotlyjs=not plotly_incluido))
            plotly_incluido = True
    pagina = f"""<!doctype html>
<html lang="es"><head><meta charset="utf-8"><title>Comparación de estrategias</title>
<style>body{{background:{FONDO};color:#e0e0e0;font-family:Segoe UI,Arial,sans-serif;margin:16px}}
h1{{font-size:20px}} h2{{font-size:18px;margin-top:32px}} p{{color:#bdbdbd}}
table{{border-collapse:collapse;font-size:14px;display:block;overflow-x:auto}}
th,td{{padding:4px 12px;border-bottom:1px solid #333;text-align:right;white-space:nowrap}}
th:first-child,td:first-child{{text-align:left}} th{{color:#bdbdbd;font-weight:600}}
summary{{cursor:pointer;color:#bdbdbd;margin:12px 0}} .pos{{color:#26a69a}} .neg{{color:#ef5350}}</style></head><body>
<h1>Comparación de estrategias · {args.periodo} {desde} a {hasta} · hora Nueva York</h1>
<p>Todos los montos son netos de comisiones y slippage. Cada estrategia parte con su propio
capital inicial. Velas de 5 minutos · triángulo = entrada, X amarilla = salida, línea roja =
stop, línea verde = TP, caja punteada = rango de apertura.</p>
{''.join(c if isinstance(c, str) else c for c in cuerpo)}
</body></html>"""
    destino = carpeta / "graficos"
    destino.mkdir(exist_ok=True)
    salida = destino / "comparacion_estrategias.html"
    salida.write_text(pagina, encoding="utf-8")
    print(f"Archivo: {salida.relative_to(RESULTADOS.parent)}")


if __name__ == "__main__":
    main()
