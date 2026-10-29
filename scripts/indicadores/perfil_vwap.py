"""Perfil de volumen y VWAP con bandas, por sesión (estilo Deepcharts). Hora de Nueva York.

Uso:
    python -m scripts.indicadores.perfil_vwap --desde 2026-09-22 --hasta 2026-09-24

Una sesión va de las 18:00 del día anterior a las 16:00 de su día (la sesión del lunes
empieza el domingo a las 18:00). Los trades de 16:00 a 18:00 no entran en ninguna sesión.
--desde y --hasta son los días en que TERMINAN las sesiones (--hasta incluido).

Todo se calcula desde cada trade de data/raw (no desde velas), así que es exacto:
- Perfil: volumen en cada precio (cada tick de 0,25). POC = precio con más volumen
  (si hay empate, el más cercano al centro del rango de la sesión). Value area = 70% del
  volumen, método clásico: desde el POC se suman los 2 precios de arriba o los 2 de
  abajo, el par que tenga más volumen, hasta llegar al 70%.
- VWAP = suma(precio x volumen) / suma(volumen), acumulado desde las 18:00.
  Bandas = VWAP ± 1 y ± 2 desviaciones estándar del precio ponderadas por volumen.
  En cada vela se muestra el valor al CIERRE de esa vela (lo que se sabía en ese momento).
El gráfico queda en resultados\\POC_VWAP\\graficos.
"""
import argparse

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from scripts.datos.construir_barras import (NY, PROCESSED, RAIZ, archivos_por_mes, leer_barras,
                                            leer_trades_mes, poner_contratos)

GRAFICOS = RAIZ / "resultados" / "POC_VWAP" / "graficos"
HORA_INICIO, HORA_FIN = 18, 16  # la sesión va de 18:00 (día anterior) a 16:00
TICK = 0.25
VALUE_AREA = 0.70
DESVIOS = (1, 2)
TEMPORALIDAD = 15
ANCHO_PERFIL = 0.35  # la fila con más volumen ocupa este % del ancho de la sesión

VERDE, ROJO = "#26a69a", "#ef5350"
COLOR_VWAP = {0: "#ffb74d", 1: "#4fc3f7", 2: "#ba68c8"}
COLOR_POC, COLOR_VA = "#ffeb3b", "#81c784"


def dia_de_sesion(ts_ny):
    """Día (sin hora) de la sesión a la que pertenece cada momento; NaT entre 16:00 y 18:00."""
    hora = ts_ny.dt.hour
    dia = ts_ny.dt.tz_localize(None).dt.normalize()
    dia = dia.where(hora < HORA_INICIO, dia + pd.Timedelta(days=1))
    return dia.where((hora < HORA_FIN) | (hora >= HORA_INICIO))


def cargar_trades(desde, hasta, saltar_rolls=False):
    """Trades de las sesiones pedidas, con su columna 'sesion'. Una sesión con roll
    adentro detiene el script, o se deja fuera si saltar_rolls=True."""
    inicio = pd.Timestamp(desde) - pd.Timedelta(days=1)
    meses = pd.period_range(inicio, hasta, freq="M").strftime("%Y-%m")
    grupos = archivos_por_mes()
    faltan = [m for m in meses if m not in grupos]
    if faltan:
        raise SystemExit(f"No hay datos crudos de: {', '.join(faltan)}")
    t = poner_contratos(leer_trades_mes([r for m in meses for r in grupos[m]]))
    t = t[t["price"].notna()].copy()
    t["ts_ny"] = t["ts_event"].dt.tz_convert(NY)
    t["sesion"] = dia_de_sesion(t["ts_ny"])
    t = t[(t["sesion"] >= pd.Timestamp(desde)) & (t["sesion"] <= pd.Timestamp(hasta))]
    if t.empty:
        raise SystemExit("No hay trades en esas sesiones.")
    for dia, s in t.groupby("sesion"):
        if s["instrument_id"].nunique() > 1:
            if not saltar_rolls:
                raise SystemExit(f"Sesión {dia:%Y-%m-%d}: hay un roll dentro de la sesión "
                                 "(mezcla dos contratos); elige otros días.")
            print(f"  Sesión {dia:%Y-%m-%d} fuera: tiene un roll adentro.")
            t = t[t["sesion"] != dia]
    return t.reset_index(drop=True)


def perfil(s):
    """Volumen por tick de una sesión + POC, VAH, VAL."""
    ticks = np.round(s["price"].to_numpy() / TICK).astype("int64")
    if np.abs(ticks * TICK - s["price"].to_numpy()).max() > 1e-6:
        raise SystemExit("Hay precios que no son múltiplo de 0,25.")
    vol = pd.Series(s["size"].to_numpy()).groupby(ticks).sum()
    vol = vol.reindex(range(vol.index.min(), vol.index.max() + 1), fill_value=0)
    v = vol.to_numpy()

    candidatos = np.flatnonzero(v == v.max())
    centro = (len(v) - 1) / 2
    poc = int(candidatos[np.argmin(np.abs(candidatos - centro))])

    abajo = arriba = poc
    acumulado, meta = v[poc], VALUE_AREA * v.sum()
    while acumulado < meta:
        par_arriba = v[arriba + 1: arriba + 3]
        par_abajo = v[max(abajo - 2, 0): abajo]
        if not len(par_arriba) and not len(par_abajo):
            break
        sube = len(par_arriba) and (not len(par_abajo) or par_arriba.sum() >= par_abajo.sum())
        baja = len(par_abajo) and (not len(par_arriba) or par_abajo.sum() >= par_arriba.sum())
        if sube:
            arriba += len(par_arriba)
            acumulado += par_arriba.sum()
        if baja:
            abajo -= len(par_abajo)
            acumulado += par_abajo.sum()
    precio = (vol.index.to_numpy() * TICK)
    return dict(precios=precio, volumen=v, poc=precio[poc], vah=precio[arriba],
                val=precio[abajo], pct_va=acumulado / v.sum(), total=int(v.sum()))


def perfil_en_desarrollo(s, momento):
    """Perfil con SOLO los trades de la sesión anteriores a 'momento' (UTC).
    REGLA INVIOLABLE (estrategias\\POC_VWAP\\REGLA_PERFIL_EN_DESARROLLO.md): para
    decidir un trade se usa esto, nunca el perfil completo de la sesión."""
    previos = s[s["ts_event"] < momento]
    if previos.empty:
        return None  # aún no hay trades en la sesión: no hay perfil
    return perfil(previos)


def poc_vwap_por_minuto(s):
    """POC en desarrollo, VWAP y σ al CIERRE de cada minuto de la sesión, usando solo los
    trades hasta ese cierre (regla inviolable). Índice = inicio del minuto, en UTC.
    El POC sigue la misma regla de empate que perfil()."""
    p, v = s["price"].to_numpy(), s["size"].to_numpy()
    ticks = np.round(p / TICK).astype("int64")
    base = ticks.min()
    idx = ticks - base
    hist = np.zeros(idx.max() + 1, dtype="int64")
    minuto = s["ts_event"].dt.floor("min").dt.tz_localize(None).to_numpy()  # UTC sin zona
    cortes = np.flatnonzero(minuto[1:] != minuto[:-1]) + 1
    inicios, fines = np.r_[0, cortes], np.r_[cortes, len(p)]

    pocs, lo, hi = [], idx[0], idx[0]
    for a, b in zip(inicios, fines):
        np.add.at(hist, idx[a:b], v[a:b])
        lo, hi = min(lo, idx[a:b].min()), max(hi, idx[a:b].max())
        tramo = hist[lo:hi + 1]
        candidatos = np.flatnonzero(tramo == tramo.max())
        pocs.append(lo + candidatos[np.argmin(np.abs(candidatos - (hi - lo) / 2))])

    ref = p[0]
    d, w = p - ref, v.astype(float)
    cv, cpv, cp2v = np.cumsum(w), np.cumsum(w * d), np.cumsum(w * d * d)
    media = cpv[fines - 1] / cv[fines - 1]
    sd = np.sqrt(np.clip(cp2v[fines - 1] / cv[fines - 1] - media ** 2, 0, None))
    return pd.DataFrame({"poc": (np.array(pocs) + base) * TICK, "vwap": ref + media, "sd": sd},
                        index=pd.DatetimeIndex(minuto[inicios]).tz_localize("UTC"))


def minutos_del_mes(mes):
    """POC en desarrollo + VWAP + vela de 1 minuto, minuto a minuto, de las sesiones que
    TERMINAN en 'mes' (AAAA-MM). Se guarda en data/processed/poc_vwap_1m_<mes>.csv y la
    próxima vez se lee de ahí. Sesiones con roll adentro, o sin el inicio de la sesión en
    los datos (la primera), quedan fuera. Controla minutos al azar contra perfil_en_desarrollo."""
    ruta = PROCESSED / f"poc_vwap_1m_{mes}.csv"
    if ruta.exists():
        r = pd.read_csv(ruta)
        r["ts_utc"] = pd.to_datetime(r["ts_utc"], utc=True, format="ISO8601")
        r["sesion"] = pd.to_datetime(r["sesion"])
        return r
    grupos = archivos_por_mes()
    previo = (pd.Period(mes, "M") - 1).strftime("%Y-%m")
    desde = pd.Timestamp(f"{mes}-01") + pd.Timedelta(days=0 if previo in grupos else 1)
    hasta = pd.Timestamp(f"{mes}-01") + pd.offsets.MonthEnd(0)
    print(f"  {mes}: calculando POC en desarrollo y VWAP desde los trades...", flush=True)
    t = cargar_trades(f"{desde:%Y-%m-%d}", f"{hasta:%Y-%m-%d}", saltar_rolls=True)
    partes, azar = [], np.random.default_rng(0)
    for n, (dia, s) in enumerate(t.groupby("sesion")):
        m = poc_vwap_por_minuto(s)
        if n < 3:
            for ts in azar.choice(m.index, 5, replace=False):
                fin = pd.Timestamp(ts) + pd.Timedelta(minutes=1)
                previos = s[s["ts_event"] < fin]
                if perfil_en_desarrollo(s, fin)["poc"] != m.loc[ts, "poc"] or abs(
                        np.average(previos["price"], weights=previos["size"]) - m.loc[ts, "vwap"]) > 1e-6:
                    raise SystemExit(f"Control del POC/VWAP en desarrollo falló en {ts}.")
        m["sesion"], m["instrument_id"] = dia, s["instrument_id"].iloc[0]
        partes.append(m)
    r = pd.concat(partes).rename_axis("ts_utc").reset_index()
    meses = {previo, mes} & set(grupos)
    barras = pd.concat(leer_barras(a) for a in sorted(PROCESSED.glob("barras_1m_*.csv"))
                       if a.stem[-7:] in meses)
    r = r.merge(barras[["ts_utc", "instrument_id", "symbol", "open", "high", "low", "close"]],
                on=["ts_utc", "instrument_id"], how="left", validate="one_to_one")
    if r["close"].isna().any():
        raise SystemExit(f"{mes}: hay minutos con trades pero sin vela de 1 minuto.")
    r.to_csv(ruta, index=False)
    return r


def vwap_por_vela(s):
    """VWAP y desviación ponderada acumulados trade a trade; valor al cierre de cada vela."""
    p, v = s["price"].to_numpy(), s["size"].to_numpy().astype(float)
    ref = p[0]  # se resta un precio de referencia para no perder precisión al elevar al cuadrado
    d = p - ref
    cv, cpv, cp2v = np.cumsum(v), np.cumsum(v * d), np.cumsum(v * d * d)
    media = cpv / cv
    sd = np.sqrt(np.clip(cp2v / cv - media ** 2, 0, None))
    tabla = pd.DataFrame({"vela": s["ts_event"].dt.floor(f"{TEMPORALIDAD}min").to_numpy(),
                          "vwap": ref + media, "sd": sd})
    return tabla.groupby("vela").last()


def verificar(t, barras, perfiles, vwaps):
    """Controles independientes; detiene el script si algo no cuadra."""
    for dia, s in t.groupby("sesion"):
        # 1) volumen de cada vela calculado desde los trades == volumen de las barras guardadas
        propio = s.groupby(s["ts_event"].dt.floor(f"{TEMPORALIDAD}min"))["size"].sum()
        guardado = barras[barras["ts_utc"].isin(propio.index)].set_index("ts_utc")["volumen"]
        if len(guardado) != len(propio) or not (propio == guardado.reindex(propio.index)).all():
            raise SystemExit(f"{dia:%Y-%m-%d}: el volumen no coincide con las barras.")
        # 2) VWAP y desviación final con fórmulas de numpy
        w = s["size"].to_numpy()
        media = np.average(s["price"], weights=w)
        sd = np.sqrt(np.average((s["price"] - media) ** 2, weights=w))
        ultimo = vwaps[dia].iloc[-1]
        if abs(ultimo["vwap"] - media) > 1e-6 or abs(ultimo["sd"] - sd) > 1e-6:
            raise SystemExit(f"{dia:%Y-%m-%d}: el VWAP no coincide con el control.")
        # 3) perfil: suma = volumen total, POC = máximo, value area >= 70%
        p = perfiles[dia]
        if p["total"] != w.sum() or p["volumen"].max() != p["volumen"][p["precios"] == p["poc"]][0]:
            raise SystemExit(f"{dia:%Y-%m-%d}: el perfil no cuadra.")
        if not (p["val"] <= p["poc"] <= p["vah"]) or p["pct_va"] < VALUE_AREA:
            raise SystemExit(f"{dia:%Y-%m-%d}: la value area no cuadra.")
        print(f"  {dia:%Y-%m-%d}: {len(s):,} trades, {p['total']:,} contratos, "
              f"volumen por vela = barras OK, VWAP = control OK, value area {p['pct_va']:.1%}")


def cargar_velas(desde, hasta):
    archivos = sorted(PROCESSED.glob(f"barras_{TEMPORALIDAD}m_*.csv"))
    b = pd.concat(leer_barras(a) for a in archivos).sort_values("ts_utc")
    b["hora_ny"] = b["ts_ny"].dt.tz_localize(None)
    inicio = pd.Timestamp(desde) - pd.Timedelta(days=1) + pd.Timedelta(hours=HORA_INICIO)
    fin = pd.Timestamp(hasta) + pd.Timedelta(hours=17)  # incluye la hora 16:00-17:00
    return b[(b["hora_ny"] >= inicio) & (b["hora_ny"] < fin)].reset_index(drop=True)


def huecos_del_eje(horas):
    todas = pd.date_range(horas.iloc[0], horas.iloc[-1], freq=f"{TEMPORALIDAD}min")
    return todas.difference(pd.DatetimeIndex(horas))


def graficar(velas, perfiles, vwaps, fila_pts):
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.82, 0.18])
    fig.add_trace(go.Candlestick(
        x=velas["hora_ny"], open=velas["open"], high=velas["high"], low=velas["low"],
        close=velas["close"], increasing_line_color=VERDE, decreasing_line_color=ROJO,
        name="NQ"), row=1, col=1)
    color_vela = [VERDE if c >= o else ROJO for o, c in zip(velas["open"], velas["close"])]
    fig.add_trace(go.Bar(x=velas["hora_ny"], y=velas["volumen"], marker_color=color_vela,
                         name="Volumen", showlegend=False), row=2, col=1)

    # VWAP: una línea por banda; un hueco (None) separa las sesiones
    lineas = {(k, signo): ([], []) for k in (0, *DESVIOS) for signo in ((1,) if k == 0 else (1, -1))}
    for dia, w in vwaps.items():
        x = w.index.tz_convert(NY).tz_localize(None)
        for (k, signo), (xs, ys) in lineas.items():
            xs.extend([*x, None])
            ys.extend([*(w["vwap"] + signo * k * w["sd"]), None])
    for (k, signo), (xs, ys) in lineas.items():
        nombre = "VWAP" if k == 0 else f"VWAP {'+' if signo > 0 else '-'}{k}σ"
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=nombre,
                                 line=dict(color=COLOR_VWAP[k], width=2 if k == 0 else 1.3),
                                 hovertemplate="%{y:,.2f}"), row=1, col=1)

    # Perfil de volumen: barras horizontales pegadas al inicio de cada sesión
    for dia, p in perfiles.items():
        x0 = dia - pd.Timedelta(days=1) + pd.Timedelta(hours=HORA_INICIO)
        x1 = dia + pd.Timedelta(hours=HORA_FIN)
        filas = pd.Series(p["volumen"]).groupby(np.floor(p["precios"] / fila_pts) * fila_pts).sum()
        ancho_max = (x1 - x0) * ANCHO_PERFIL
        for precio, vol in filas.items():
            en_va = precio + fila_pts > p["val"] and precio <= p["vah"]
            fig.add_shape(type="rect", xref="x", yref="y", layer="below", line_width=0,
                          x0=x0, x1=x0 + ancho_max * (vol / filas.max()),
                          y0=precio, y1=precio + fila_pts,
                          fillcolor="rgba(92,124,250,0.45)" if en_va else "rgba(158,158,158,0.28)")
        fig.add_shape(type="rect", xref="x", yref="y", layer="below", line_width=0,
                      x0=x0, x1=x1, y0=p["val"], y1=p["vah"], fillcolor="rgba(129,199,132,0.06)")
        for nivel, color, texto, guion in ((p["poc"], COLOR_POC, "POC", "solid"),
                                          (p["vah"], COLOR_VA, "VAH", "dash"),
                                          (p["val"], COLOR_VA, "VAL", "dash")):
            fig.add_shape(type="line", xref="x", yref="y", x0=x0, x1=x1, y0=nivel, y1=nivel,
                          line=dict(color=color, width=1.6, dash=guion))
            fig.add_annotation(x=x1, y=nivel, xref="x", yref="y", xanchor="left",
                               text=f"{texto} {nivel:,.2f}", showarrow=False,
                               font=dict(color=color, size=10))

    fig.update_xaxes(rangebreaks=[dict(values=huecos_del_eje(velas["hora_ny"]),
                                       dvalue=TEMPORALIDAD * 60_000)])
    fig.update_layout(
        title=f"NQ {TEMPORALIDAD} min · perfil de volumen + VWAP por sesión (18:00–16:00 NY) · "
              f"{min(perfiles):%Y-%m-%d} a {max(perfiles):%Y-%m-%d}",
        xaxis_rangeslider_visible=False, template="plotly_dark", hovermode="x unified",
        height=900, legend=dict(orientation="h", y=1.03, x=0),
    )
    return fig


def tabla_html(perfiles, vwaps):
    filas = []
    for dia, p in perfiles.items():
        w = vwaps[dia].iloc[-1]
        filas.append(
            f"<tr><td>{dia:%a %d-%m-%Y}</td><td>{p['total']:,}</td><td>{p['poc']:,.2f}</td>"
            f"<td>{p['vah']:,.2f}</td><td>{p['val']:,.2f}</td><td>{p['vah'] - p['val']:,.2f}</td>"
            f"<td>{p['pct_va']:.1%}</td><td>{w['vwap']:,.2f}</td><td>{w['sd']:,.2f}</td></tr>")
    return (
        "<h3>Valores de cada sesión (al cierre de las 16:00)</h3><table><tr><th>Sesión</th>"
        "<th>Volumen</th><th>POC</th><th>VAH</th><th>VAL</th><th>Ancho VA (pts)</th>"
        "<th>% en VA</th><th>VWAP final</th><th>1σ (pts)</th></tr>" + "".join(filas) +
        "</table><p>Calculado trade a trade desde los datos crudos de Databento. Perfil en "
        "ticks de 0,25 (en el gráfico se agrupa en filas para que se vea mejor; POC, VAH y "
        "VAL son exactos al tick). Value area 70%, método clásico de pares. Bandas: ±1σ y "
        "±2σ ponderadas por volumen. Los trades de 16:00 a 18:00 no entran en ninguna sesión.</p>")


def main():
    parser = argparse.ArgumentParser(description="Perfil de volumen + VWAP por sesión.")
    parser.add_argument("--desde", default="2026-09-22", help="Primera sesión (día en que termina)")
    parser.add_argument("--hasta", default="2026-09-24", help="Última sesión (incluida)")
    parser.add_argument("--fila", type=float, default=5.0,
                        help="Puntos por fila del perfil en el dibujo (el cálculo es por tick)")
    args = parser.parse_args()

    print("Leyendo trades crudos...")
    t = cargar_trades(args.desde, args.hasta)
    perfiles = {dia: perfil(s) for dia, s in t.groupby("sesion")}
    vwaps = {dia: vwap_por_vela(s) for dia, s in t.groupby("sesion")}
    barras = pd.concat(leer_barras(a) for a in sorted(PROCESSED.glob(f"barras_{TEMPORALIDAD}m_*.csv")))
    print("Controles:")
    verificar(t, barras, perfiles, vwaps)

    velas = cargar_velas(args.desde, args.hasta)
    fig = graficar(velas, perfiles, vwaps, args.fila)
    GRAFICOS.mkdir(parents=True, exist_ok=True)
    salida = GRAFICOS / f"perfil_vwap_{TEMPORALIDAD}m_{args.desde}_{args.hasta}.html"
    estilo = ("<style>body{background:#111;color:#ddd;font-family:sans-serif;margin:16px}"
              "table{border-collapse:collapse}td,th{border:1px solid #444;padding:4px 10px;"
              "text-align:right}</style>")
    salida.write_text(f"<html><head><meta charset='utf-8'>{estilo}</head><body>"
                      + fig.to_html(full_html=False, include_plotlyjs=True)
                      + tabla_html(perfiles, vwaps) + "</body></html>", encoding="utf-8")
    print(f"Gráfico guardado en {salida.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
